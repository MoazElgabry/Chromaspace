#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

std::filesystem::path executableDirectory() {
#if defined(_WIN32)
  wchar_t buffer[MAX_PATH] = {};
  const DWORD len = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
  if (len == 0 || len >= static_cast<DWORD>(std::size(buffer))) return {};
  return std::filesystem::path(buffer).parent_path();
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> buffer(size + 1, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};
  std::error_code ec;
  return std::filesystem::canonical(buffer.data(), ec).parent_path();
#else
  std::vector<char> buffer(4096, '\0');
  const ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (len <= 0) return {};
  buffer[static_cast<size_t>(len)] = '\0';
  return std::filesystem::path(buffer.data()).parent_path();
#endif
}

std::filesystem::path bundledBridgeScript() {
  const std::filesystem::path exeDir = executableDirectory();
  if (exeDir.empty()) return {};
  const std::filesystem::path candidates[] = {
      exeDir.parent_path() / "Resources" / "chromaspace_resolve_bridge.py",
      exeDir / "chromaspace_resolve_bridge.py",
  };
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(candidate, ec)) return candidate;
  }
  return {};
}

#if defined(_WIN32)
std::wstring findProgramOnPath(const wchar_t* name) {
  wchar_t buffer[MAX_PATH] = {};
  const DWORD len = SearchPathW(nullptr, name, nullptr, static_cast<DWORD>(std::size(buffer)), buffer, nullptr);
  if (len == 0 || len >= static_cast<DWORD>(std::size(buffer))) return {};
  return std::wstring(buffer, buffer + len);
}

std::wstring quote(const std::wstring& text) {
  return L"\"" + text + L"\"";
}

bool launchAndWaitWindows(const std::filesystem::path& scriptPath) {
  SetEnvironmentVariableW(L"CHROMASPACE_BRIDGE_PROCESS_NAME", L"Chromaspace Resolve Bridge");
  const std::wstring script = scriptPath.wstring();
  struct Candidate {
    std::wstring exe;
    std::wstring argsPrefix;
  };
  const Candidate candidates[] = {
      {findProgramOnPath(L"pythonw.exe"), L""},
      {findProgramOnPath(L"pyw.exe"), L"-3 "},
      {findProgramOnPath(L"python.exe"), L""},
      {findProgramOnPath(L"py.exe"), L"-3 "},
  };
  for (const auto& candidate : candidates) {
    if (candidate.exe.empty()) continue;
    std::wstring command = quote(candidate.exe) + L" " + candidate.argsPrefix + quote(script) +
                           L" --chromaspace-resolve-bridge-worker";
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr,
                                   command.data(),
                                   nullptr,
                                   nullptr,
                                   FALSE,
                                   CREATE_NO_WINDOW,
                                   nullptr,
                                   nullptr,
                                   &si,
                                   &pi);
    if (!ok) continue;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return exitCode == 0;
  }
  return false;
}
#else
bool launchAndWaitPosix(const std::filesystem::path& scriptPath) {
  setenv("CHROMASPACE_BRIDGE_PROCESS_NAME", "Chromaspace Resolve Bridge", 1);
  const std::string script = scriptPath.string();
  const char* candidates[] = {"python3", "python"};
  for (const char* candidate : candidates) {
    const pid_t pid = fork();
    if (pid < 0) continue;
    if (pid == 0) {
      execlp(candidate,
             candidate,
             script.c_str(),
             "--chromaspace-resolve-bridge-worker",
             static_cast<char*>(nullptr));
      _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 127) continue;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
  return false;
}
#endif

}  // namespace

int main() {
  const std::filesystem::path script = bundledBridgeScript();
  if (script.empty()) {
    std::cerr << "Chromaspace Resolve bridge script was not found.\n";
    return 2;
  }
#if defined(_WIN32)
  return launchAndWaitWindows(script) ? 0 : 1;
#else
  return launchAndWaitPosix(script) ? 0 : 1;
#endif
}
