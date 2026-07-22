#import <Foundation/Foundation.h>

#include "ChromaspaceSourceExchangeManagerCore.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <pwd.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr const char* kAgentPlistName =
    "com.chromaspace.SourceExchangeBroker.plist";
constexpr const char* kBrokerExecutable =
    "com.chromaspace.SourceExchangeBroker";
constexpr const char* kBrokerService =
    "com.chromaspace.SourceExchangeBroker";
constexpr const char* kBootstrapService =
    "com.chromaspace.SourceExchangeBootstrap";
constexpr const char* kTemplateBrokerPath =
    "__CHROMASPACE_SOURCE_EXCHANGE_BROKER_EXECUTABLE__";
constexpr off_t kMaximumManagedFileBytes =
    static_cast<off_t>(256) * 1024 * 1024;

std::string text(NSString* value) {
  if (value == nil) return {};
  const char* utf8 = value.UTF8String;
  return utf8 ? utf8 : "";
}

std::string posixError(const char* operation) {
  return std::string(operation) + ":" + std::strerror(errno);
}

bool fail(std::string* error, const std::string& reason) {
  if (error) *error = reason;
  return false;
}

struct UserPaths {
  uid_t uid = 0;
  NSString* home = nil;
  NSString* runtimeDirectory = nil;
  NSString* broker = nil;
  NSString* launchAgents = nil;
  NSString* plist = nil;
  NSString* domain = nil;
  NSString* serviceTarget = nil;
};

bool currentUserPaths(UserPaths* paths, std::string* error) {
  if (error) error->clear();
  if (paths == nullptr) return fail(error, "missing-user-paths");
  const uid_t uid = geteuid();
  if (uid == 0) return fail(error, "root-execution-forbidden");

  long capacity = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (capacity < 1024) capacity = 16384;
  std::vector<char> storage(static_cast<size_t>(capacity));
  passwd record {};
  passwd* resolved = nullptr;
  const int lookup =
      getpwuid_r(uid, &record, storage.data(), storage.size(), &resolved);
  if (lookup != 0 || resolved == nullptr || record.pw_dir == nullptr ||
      record.pw_dir[0] != '/') {
    return fail(error, "effective-user-home-unavailable");
  }

  NSString* home =
      [[NSString stringWithUTF8String:record.pw_dir]
          stringByStandardizingPath];
  if (home.length == 0 || [home isEqualToString:@"/"]) {
    return fail(error, "effective-user-home-invalid");
  }
  NSString* applicationSupport =
      [home stringByAppendingPathComponent:@"Library/Application Support"];
  NSString* runtime =
      [applicationSupport
          stringByAppendingPathComponent:@"Chromaspace/SourceExchangeV2"];
  NSString* launchAgents =
      [home stringByAppendingPathComponent:@"Library/LaunchAgents"];
  paths->uid = uid;
  paths->home = home;
  paths->runtimeDirectory = runtime;
  paths->broker =
      [runtime stringByAppendingPathComponent:
                   [NSString stringWithUTF8String:kBrokerExecutable]];
  paths->launchAgents = launchAgents;
  paths->plist =
      [launchAgents
          stringByAppendingPathComponent:
              [NSString stringWithUTF8String:kAgentPlistName]];
  paths->domain =
      [NSString stringWithFormat:@"gui/%u", static_cast<unsigned>(uid)];
  paths->serviceTarget =
      [NSString stringWithFormat:@"gui/%u/%s",
                                 static_cast<unsigned>(uid),
                                 kBrokerService];
  return true;
}

bool isWithinHome(NSString* home, NSString* path) {
  NSString* canonicalHome = home.stringByStandardizingPath;
  NSString* canonicalPath = path.stringByStandardizingPath;
  return [canonicalPath isEqualToString:canonicalHome] ||
         [canonicalPath
             hasPrefix:[canonicalHome stringByAppendingString:@"/"]];
}

bool validateOwnedNode(NSString* path,
                       uid_t uid,
                       mode_t expectedType,
                       bool allowMissing,
                       std::string* error,
                       struct stat* observed = nullptr) {
  struct stat value {};
  if (lstat(path.fileSystemRepresentation, &value) != 0) {
    if (allowMissing && errno == ENOENT) return true;
    return fail(error, posixError("lstat"));
  }
  if (S_ISLNK(value.st_mode)) {
    return fail(error, "symlink-forbidden:" + text(path));
  }
  if ((value.st_mode & S_IFMT) != expectedType) {
    return fail(error, "unexpected-file-type:" + text(path));
  }
  if (value.st_uid != uid) {
    return fail(error, "foreign-owner:" + text(path));
  }
  if (observed) *observed = value;
  return true;
}

bool validateOwnedPathFromHome(NSString* home,
                               NSString* path,
                               uid_t uid,
                               std::string* error) {
  if (!isWithinHome(home, path)) {
    return fail(error, "path-outside-effective-user-home");
  }
  if (!validateOwnedNode(home, uid, S_IFDIR, false, error)) return false;
  NSString* relative =
      [path substringFromIndex:
                [home isEqualToString:@"/"] ? 1 : home.length + 1];
  NSString* current = home;
  for (NSString* component in relative.pathComponents) {
    if ([component isEqualToString:@"/"] ||
        [component isEqualToString:@"."] || component.length == 0) {
      continue;
    }
    if ([component isEqualToString:@".."]) {
      return fail(error, "parent-path-component-forbidden");
    }
    current = [current stringByAppendingPathComponent:component];
    struct stat value {};
    if (lstat(current.fileSystemRepresentation, &value) != 0) {
      return fail(error, posixError("lstat-path-component"));
    }
    if (S_ISLNK(value.st_mode) || value.st_uid != uid) {
      return fail(error, "unsafe-path-component:" + text(current));
    }
  }
  return true;
}

bool validateOwnedExistingPrefixFromHome(NSString* home,
                                         NSString* path,
                                         uid_t uid,
                                         std::string* error) {
  if (!isWithinHome(home, path)) {
    return fail(error, "path-outside-effective-user-home");
  }
  if (!validateOwnedNode(home, uid, S_IFDIR, false, error)) return false;
  NSString* relative = [path substringFromIndex:home.length + 1];
  NSString* current = home;
  for (NSString* component in relative.pathComponents) {
    if ([component isEqualToString:@"/"] ||
        [component isEqualToString:@"."] || component.length == 0) {
      continue;
    }
    current = [current stringByAppendingPathComponent:component];
    struct stat value {};
    if (lstat(current.fileSystemRepresentation, &value) != 0) {
      if (errno == ENOENT) return true;
      return fail(error, posixError("lstat-path-component"));
    }
    if (S_ISLNK(value.st_mode) || value.st_uid != uid) {
      return fail(error, "unsafe-path-component:" + text(current));
    }
  }
  return true;
}

bool ensureOwnedDirectory(NSString* path,
                          uid_t uid,
                          mode_t mode,
                          bool enforceMode,
                          std::string* error) {
  struct stat value {};
  if (lstat(path.fileSystemRepresentation, &value) != 0) {
    if (errno != ENOENT) return fail(error, posixError("lstat-directory"));
    if (mkdir(path.fileSystemRepresentation, mode) != 0) {
      return fail(error, posixError("mkdir"));
    }
    if (lstat(path.fileSystemRepresentation, &value) != 0) {
      return fail(error, posixError("lstat-created-directory"));
    }
  }
  if (S_ISLNK(value.st_mode) || !S_ISDIR(value.st_mode) ||
      value.st_uid != uid) {
    return fail(error, "unsafe-directory:" + text(path));
  }
  if (enforceMode && (value.st_mode & 0777) != mode &&
      chmod(path.fileSystemRepresentation, mode) != 0) {
    return fail(error, posixError("chmod-directory"));
  }
  return true;
}

bool ensureDestinationDirectories(const UserPaths& paths,
                                  std::string* error) {
  if (!validateOwnedNode(paths.home, paths.uid, S_IFDIR, false, error)) {
    return false;
  }
  NSString* library =
      [paths.home stringByAppendingPathComponent:@"Library"];
  NSString* applicationSupport =
      [library stringByAppendingPathComponent:@"Application Support"];
  NSString* chromaspace =
      [applicationSupport stringByAppendingPathComponent:@"Chromaspace"];
  if (!ensureOwnedDirectory(library, paths.uid, 0700, false, error) ||
      !ensureOwnedDirectory(
          applicationSupport, paths.uid, 0700, false, error) ||
      !ensureOwnedDirectory(chromaspace, paths.uid, 0700, true, error) ||
      !ensureOwnedDirectory(
          paths.runtimeDirectory, paths.uid, 0700, true, error) ||
      !ensureOwnedDirectory(
          paths.launchAgents, paths.uid, 0700, false, error)) {
    return false;
  }
  return true;
}

bool readOwnedFile(NSString* path,
                   uid_t uid,
                   bool executable,
                   NSData** data,
                   struct stat* metadata,
                   std::string* error,
                   bool allowEmpty = false) {
  const int fd = open(path.fileSystemRepresentation, O_RDONLY | O_NOFOLLOW);
  if (fd < 0) return fail(error, posixError("open-source"));
  struct stat value {};
  bool valid =
      fstat(fd, &value) == 0 && S_ISREG(value.st_mode) &&
      value.st_uid == uid && (allowEmpty || value.st_size > 0) &&
      value.st_size <= kMaximumManagedFileBytes &&
      (!executable || (value.st_mode & S_IXUSR) != 0);
  if (!valid) {
    close(fd);
    return fail(error, "source-file-contract-invalid:" + text(path));
  }
  NSMutableData* bytes = [NSMutableData data];
  unsigned char buffer[65536];
  for (;;) {
    const ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count == 0) break;
    if (count < 0) {
      if (errno == EINTR) continue;
      close(fd);
      return fail(error, posixError("read-source"));
    }
    [bytes appendBytes:buffer length:static_cast<NSUInteger>(count)];
  }
  close(fd);
  if (!allowEmpty && bytes.length == 0) {
    return fail(error, "source-file-empty");
  }
  if (data) *data = bytes;
  if (metadata) *metadata = value;
  return true;
}

bool destinationFileState(NSString* path,
                          uid_t uid,
                          mode_t requiredMode,
                          NSData** data,
                          bool* exists,
                          std::string* error) {
  if (exists) *exists = false;
  struct stat value {};
  if (lstat(path.fileSystemRepresentation, &value) != 0) {
    if (errno == ENOENT) return true;
    return fail(error, posixError("lstat-destination"));
  }
  if (S_ISLNK(value.st_mode) || !S_ISREG(value.st_mode) ||
      value.st_uid != uid) {
    return fail(error, "unsafe-destination:" + text(path));
  }
  if (exists) *exists = true;
  if ((value.st_mode & 0777) != requiredMode ||
      value.st_size < 0 || value.st_size > kMaximumManagedFileBytes ||
      data == nullptr) {
    return true;
  }
  return readOwnedFile(
      path, uid, false, data, nullptr, error, true);
}

bool writeAll(int fd, const void* bytes, size_t length) {
  const unsigned char* cursor =
      static_cast<const unsigned char*>(bytes);
  while (length > 0) {
    const ssize_t count = write(fd, cursor, length);
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (count == 0) return false;
    cursor += count;
    length -= static_cast<size_t>(count);
  }
  return true;
}

bool atomicReplace(NSString* destination,
                   NSData* data,
                   uid_t uid,
                   mode_t mode,
                   std::string* error) {
  bool exists = false;
  if (!destinationFileState(
          destination, uid, mode, nullptr, &exists, error)) {
    return false;
  }
  (void)exists;
  NSString* directory = destination.stringByDeletingLastPathComponent;
  NSString* pattern =
      [directory stringByAppendingPathComponent:@".chromaspace-install.XXXXXX"];
  std::vector<char> writable(
      [pattern lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
  if (![pattern getCString:writable.data()
                maxLength:writable.size()
                 encoding:NSUTF8StringEncoding]) {
    return fail(error, "temporary-path-encoding-failed");
  }
  const int fd = mkstemp(writable.data());
  if (fd < 0) return fail(error, posixError("mkstemp"));
  const std::string temporary(writable.data());
  bool written =
      fchmod(fd, mode) == 0 &&
      writeAll(fd, data.bytes, static_cast<size_t>(data.length)) &&
      fsync(fd) == 0;
  const int closeResult = close(fd);
  if (!written || closeResult != 0) {
    unlink(temporary.c_str());
    return fail(error, posixError("write-temporary"));
  }
  if (rename(temporary.c_str(), destination.fileSystemRepresentation) != 0) {
    unlink(temporary.c_str());
    return fail(error, posixError("rename-temporary"));
  }
  const int directoryFd =
      open(directory.fileSystemRepresentation, O_RDONLY | O_DIRECTORY);
  if (directoryFd >= 0) {
    fsync(directoryFd);
    close(directoryFd);
  }
  return true;
}

NSDictionary* launchAgentDictionary(NSString* brokerPath) {
  return @{
    @"Label" : [NSString stringWithUTF8String:kBrokerService],
    @"ProgramArguments" : @[ brokerPath ],
    @"MachServices" : @{
      [NSString stringWithUTF8String:kBrokerService] : @YES,
      [NSString stringWithUTF8String:kBootstrapService] : @YES,
    },
    @"ProcessType" : @"Background",
    @"ThrottleInterval" : @5,
  };
}

bool exactPlist(NSDictionary* value, NSString* brokerPath) {
  return value != nil &&
         [value isEqualToDictionary:launchAgentDictionary(brokerPath)];
}

bool plistData(NSDictionary* value, NSData** data, std::string* error) {
  NSError* plistError = nil;
  NSData* encoded =
      [NSPropertyListSerialization
          dataWithPropertyList:value
                        format:NSPropertyListXMLFormat_v1_0
                       options:0
                         error:&plistError];
  if (encoded == nil) {
    return fail(error,
                "plist-serialization-failed:" +
                    text(plistError.localizedDescription));
  }
  if (data) *data = encoded;
  return true;
}

bool parsePlist(NSData* data, NSDictionary** value, std::string* error) {
  NSError* plistError = nil;
  id parsed =
      [NSPropertyListSerialization
          propertyListWithData:data
                       options:NSPropertyListImmutable
                        format:nullptr
                         error:&plistError];
  if (![parsed isKindOfClass:[NSDictionary class]]) {
    return fail(error,
                "plist-parse-failed:" +
                    text(plistError.localizedDescription));
  }
  if (value) *value = (NSDictionary*)parsed;
  return true;
}

bool runLaunchctl(NSArray<NSString*>* arguments,
                  int* exitCode,
                  std::string* error) {
  NSTask* task = [[NSTask alloc] init];
  task.executableURL = [NSURL fileURLWithPath:@"/bin/launchctl"];
  task.arguments = arguments;
  task.standardOutput = NSFileHandle.fileHandleWithNullDevice;
  task.standardError = NSFileHandle.fileHandleWithNullDevice;
  NSError* launchError = nil;
  if (![task launchAndReturnError:&launchError]) {
    return fail(error,
                "launchctl-start-failed:" +
                    text(launchError.localizedDescription));
  }
  [task waitUntilExit];
  if (exitCode) *exitCode = task.terminationStatus;
  return true;
}

bool serviceIsLoaded(const UserPaths& paths,
                     bool* loaded,
                     std::string* error) {
  int exitCode = 1;
  if (!runLaunchctl(@[ @"print", paths.serviceTarget ],
                    &exitCode,
                    error)) {
    return false;
  }
  if (loaded) *loaded = exitCode == 0;
  return true;
}

bool bootoutIfLoaded(const UserPaths& paths, std::string* error) {
  bool loaded = false;
  if (!serviceIsLoaded(paths, &loaded, error)) return false;
  if (!loaded) return true;
  int exitCode = 1;
  if (!runLaunchctl(@[ @"bootout", paths.serviceTarget ],
                    &exitCode,
                    error)) {
    return false;
  }
  if (exitCode != 0) return fail(error, "launchctl-bootout-failed");
  return true;
}

class BundleValidator final
    : public ChromaspaceSourceExchange::ManagedBundleValidator {
 public:
  bool validate(std::string* error) override {
    if (error) error->clear();
    UserPaths paths;
    if (!currentUserPaths(&paths, error)) return false;

    NSBundle* bundle = NSBundle.mainBundle;
    NSString* bundlePath = bundle.bundlePath.stringByStandardizingPath;
    NSString* broker =
        [bundlePath
            stringByAppendingPathComponent:
                [@"Contents/MacOS/"
                    stringByAppendingString:
                        [NSString stringWithUTF8String:kBrokerExecutable]]];
    NSString* agent =
        [bundlePath
            stringByAppendingPathComponent:
                [@"Contents/Library/LaunchAgents/"
                    stringByAppendingString:
                        [NSString stringWithUTF8String:kAgentPlistName]]];
    if (bundlePath.length == 0 ||
        !validateOwnedPathFromHome(
            paths.home, broker, paths.uid, error) ||
        !validateOwnedPathFromHome(
            paths.home, agent, paths.uid, error)) {
      return false;
    }

    NSData* brokerData = nil;
    NSData* agentData = nil;
    if (!readOwnedFile(
            broker, paths.uid, true, &brokerData, nullptr, error) ||
        !readOwnedFile(
            agent, paths.uid, false, &agentData, nullptr, error)) {
      return false;
    }
    NSDictionary* plist = nil;
    if (!parsePlist(agentData, &plist, error)) return false;
    return exactPlist(
               plist,
               [NSString stringWithUTF8String:kTemplateBrokerPath]) ||
           fail(error, "launch-agent-template-contract-invalid");
  }
};

class LaunchAgentAdapter final
    : public ChromaspaceSourceExchange::ManagedServiceAdapter {
 public:
  explicit LaunchAgentAdapter(std::string* error) {
    valid_ = currentUserPaths(&paths_, error);
    if (!valid_) return;
    NSString* bundlePath =
        NSBundle.mainBundle.bundlePath.stringByStandardizingPath;
    sourceBroker_ =
        [bundlePath
            stringByAppendingPathComponent:
                [@"Contents/MacOS/"
                    stringByAppendingString:
                        [NSString stringWithUTF8String:kBrokerExecutable]]];
  }

  ChromaspaceSourceExchange::ManagedServiceStatus status(
      std::string* error) override {
    if (error) error->clear();
    if (!valid_) {
      if (error) *error = "effective-user-context-invalid";
      return ChromaspaceSourceExchange::ManagedServiceStatus::Unknown;
    }
    if (!validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.runtimeDirectory, paths_.uid, error) ||
        !validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.launchAgents, paths_.uid, error)) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::Unknown;
    }
    bool brokerExists = false;
    bool plistExists = false;
    NSData* brokerData = nil;
    NSData* installedPlistData = nil;
    if (!destinationFileState(paths_.broker,
                              paths_.uid,
                              0500,
                              &brokerData,
                              &brokerExists,
                              error) ||
        !destinationFileState(paths_.plist,
                              paths_.uid,
                              0600,
                              &installedPlistData,
                              &plistExists,
                              error)) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::Unknown;
    }
    if (!brokerExists || !plistExists || brokerData == nil ||
        installedPlistData == nil) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::NotRegistered;
    }
    NSDictionary* installedPlist = nil;
    if (!parsePlist(installedPlistData, &installedPlist, nullptr)) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::
          NotRegistered;
    }
    if (!exactPlist(installedPlist, paths_.broker)) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::NotRegistered;
    }
    bool loaded = false;
    if (!serviceIsLoaded(paths_, &loaded, error)) {
      return ChromaspaceSourceExchange::ManagedServiceStatus::Unknown;
    }
    return loaded
               ? ChromaspaceSourceExchange::ManagedServiceStatus::Enabled
               : ChromaspaceSourceExchange::ManagedServiceStatus::
                     NotRegistered;
  }

  bool registerService(std::string* error) override {
    if (error) error->clear();
    if (!valid_) return fail(error, "effective-user-context-invalid");
    if (!validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.runtimeDirectory, paths_.uid, error) ||
        !validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.launchAgents, paths_.uid, error)) {
      return false;
    }
    if (!ensureDestinationDirectories(paths_, error)) return false;

    NSData* source = nil;
    if (!readOwnedFile(
            sourceBroker_, paths_.uid, true, &source, nullptr, error)) {
      return false;
    }
    NSData* desiredPlist = nil;
    if (!plistData(
            launchAgentDictionary(paths_.broker), &desiredPlist, error)) {
      return false;
    }

    bool brokerExists = false;
    bool plistExists = false;
    NSData* installedBroker = nil;
    NSData* installedPlist = nil;
    if (!destinationFileState(paths_.broker,
                              paths_.uid,
                              0500,
                              &installedBroker,
                              &brokerExists,
                              error) ||
        !destinationFileState(paths_.plist,
                              paths_.uid,
                              0600,
                              &installedPlist,
                              &plistExists,
                              error)) {
      return false;
    }
    bool loaded = false;
    if (!serviceIsLoaded(paths_, &loaded, error)) return false;
    if (loaded && brokerExists && plistExists &&
        [source isEqualToData:installedBroker]) {
      NSDictionary* installedDictionary = nil;
      if (installedPlist != nil &&
          parsePlist(installedPlist, &installedDictionary, nullptr) &&
          exactPlist(installedDictionary, paths_.broker)) {
        return true;
      }
    }

    if (!bootoutIfLoaded(paths_, error) ||
        !atomicReplace(paths_.broker, source, paths_.uid, 0500, error) ||
        !atomicReplace(
            paths_.plist, desiredPlist, paths_.uid, 0600, error)) {
      return false;
    }
    int exitCode = 1;
    if (!runLaunchctl(
            @[ @"bootstrap", paths_.domain, paths_.plist ],
            &exitCode,
            error)) {
      return false;
    }
    return exitCode == 0 ||
           fail(error, "launchctl-bootstrap-failed");
  }

  bool unregisterService(std::string* error) override {
    if (error) error->clear();
    if (!valid_) return fail(error, "effective-user-context-invalid");

    if (!validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.runtimeDirectory, paths_.uid, error) ||
        !validateOwnedExistingPrefixFromHome(
            paths_.home, paths_.launchAgents, paths_.uid, error)) {
      return false;
    }

    bool brokerExists = false;
    bool plistExists = false;
    if (!destinationFileState(paths_.broker,
                              paths_.uid,
                              0500,
                              nullptr,
                              &brokerExists,
                              error) ||
        !destinationFileState(paths_.plist,
                              paths_.uid,
                              0600,
                              nullptr,
                              &plistExists,
                              error) ||
        !bootoutIfLoaded(paths_, error)) {
      return false;
    }
    if (plistExists &&
        unlink(paths_.plist.fileSystemRepresentation) != 0) {
      return fail(error, posixError("unlink-launch-agent"));
    }
    if (brokerExists &&
        unlink(paths_.broker.fileSystemRepresentation) != 0) {
      return fail(error, posixError("unlink-broker"));
    }
    if (rmdir(paths_.runtimeDirectory.fileSystemRepresentation) != 0 &&
        errno != ENOENT && errno != ENOTEMPTY) {
      return fail(error, posixError("rmdir-runtime-directory"));
    }
    return true;
  }

 private:
  bool valid_ = false;
  UserPaths paths_;
  NSString* sourceBroker_ = nil;
};

int emitResult(
    const ChromaspaceSourceExchange::SourceExchangeManagerResult& result) {
  std::fprintf(result.exitCode == 0 ? stdout : stderr,
               "%s\n",
               result.output.c_str());
  return result.exitCode;
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 2 || argv[1] == nullptr) {
      std::fprintf(
          stderr,
          "status=invalid reason=usage-register-unregister-status-validate\n");
      return 2;
    }
    const std::string command = argv[1];
    if (command == "validate") {
      BundleValidator validator;
      return emitResult(
          ChromaspaceSourceExchange::runSourceExchangeManagerCommand(
              command, nullptr, &validator));
    }
    std::string constructionError;
    LaunchAgentAdapter service(&constructionError);
    if (!constructionError.empty()) {
      std::fprintf(stderr,
                   "status=error reason=%s\n",
                   constructionError.c_str());
      return 1;
    }
    if (command == "unregister") {
      return emitResult(
          ChromaspaceSourceExchange::runSourceExchangeManagerCommand(
              command, &service, nullptr));
    }
    if (command == "register" || command == "status") {
      BundleValidator validator;
      return emitResult(
          ChromaspaceSourceExchange::runSourceExchangeManagerCommand(
              command, &service, &validator));
    }
    return emitResult(
        ChromaspaceSourceExchange::runSourceExchangeManagerCommand(
            command, nullptr, nullptr));
  }
}
