#include "ChromaspaceViewerWorkspacePersistence.h"

#include "ChromaspaceViewerLayout.h"
#include "ChromaspaceViewerState.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ChromaspaceViewer {
namespace {

constexpr std::size_t kMaxJsonDepth = 24u;
constexpr std::size_t kMaxJsonMembers = 256u;
constexpr std::size_t kMaxJsonArrayItems = kViewerWorkspaceMaxLassoPoints + 64u;
constexpr float kRectEpsilon = 1.0e-5f;

struct JsonValue {
  enum class Type : uint8_t { Null, Bool, Number, String, Object, Array };
  Type type = Type::Null;
  bool boolean = false;
  std::string number;
  std::string string;
  std::vector<std::pair<std::string, JsonValue>> object;
  std::vector<JsonValue> array;
};

bool finite(float value) noexcept { return std::isfinite(value); }
bool finite(double value) noexcept { return std::isfinite(value); }

bool isHex(char c) noexcept {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

unsigned hexValue(char c) noexcept {
  if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
  if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
  return static_cast<unsigned>(c - 'A' + 10);
}

bool appendUtf8(std::string* out, unsigned codePoint) {
  if (!out || codePoint > 0x10ffffu ||
      (codePoint >= 0xd800u && codePoint <= 0xdfffu)) {
    return false;
  }
  if (codePoint <= 0x7fu) {
    out->push_back(static_cast<char>(codePoint));
  } else if (codePoint <= 0x7ffu) {
    out->push_back(static_cast<char>(0xc0u | (codePoint >> 6u)));
    out->push_back(static_cast<char>(0x80u | (codePoint & 0x3fu)));
  } else if (codePoint <= 0xffffu) {
    out->push_back(static_cast<char>(0xe0u | (codePoint >> 12u)));
    out->push_back(static_cast<char>(0x80u | ((codePoint >> 6u) & 0x3fu)));
    out->push_back(static_cast<char>(0x80u | (codePoint & 0x3fu)));
  } else {
    out->push_back(static_cast<char>(0xf0u | (codePoint >> 18u)));
    out->push_back(static_cast<char>(0x80u | ((codePoint >> 12u) & 0x3fu)));
    out->push_back(static_cast<char>(0x80u | ((codePoint >> 6u) & 0x3fu)));
    out->push_back(static_cast<char>(0x80u | (codePoint & 0x3fu)));
  }
  return true;
}

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  bool parse(JsonValue* value) {
    if (!value) return false;
    skipWhitespace();
    if (!parseValue(value, 0u)) return false;
    skipWhitespace();
    return position_ == input_.size();
  }

 private:
  bool parseValue(JsonValue* value, std::size_t depth) {
    if (!value || depth > kMaxJsonDepth || position_ >= input_.size()) return false;
    const char c = input_[position_];
    if (c == 'n') return parseLiteral("null", JsonValue::Type::Null, value);
    if (c == 't') {
      if (!parseLiteral("true", JsonValue::Type::Bool, value)) return false;
      value->boolean = true;
      return true;
    }
    if (c == 'f') {
      if (!parseLiteral("false", JsonValue::Type::Bool, value)) return false;
      value->boolean = false;
      return true;
    }
    if (c == '"') {
      value->type = JsonValue::Type::String;
      return parseString(&value->string);
    }
    if (c == '{') return parseObject(value, depth + 1u);
    if (c == '[') return parseArray(value, depth + 1u);
    value->type = JsonValue::Type::Number;
    return parseNumber(&value->number);
  }

  bool parseLiteral(std::string_view literal, JsonValue::Type type,
                    JsonValue* value) {
    if (input_.substr(position_, literal.size()) != literal) return false;
    position_ += literal.size();
    value->type = type;
    return true;
  }

  bool parseString(std::string* out) {
    if (!out || position_ >= input_.size() || input_[position_] != '"') return false;
    ++position_;
    out->clear();
    while (position_ < input_.size()) {
      const unsigned char c = static_cast<unsigned char>(input_[position_++]);
      if (c == '"') return true;
      if (c < 0x20u) return false;
      if (c != '\\') {
        out->push_back(static_cast<char>(c));
        continue;
      }
      if (position_ >= input_.size()) return false;
      const char escape = input_[position_++];
      switch (escape) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          if (position_ + 4u > input_.size()) return false;
          unsigned codePoint = 0u;
          for (unsigned i = 0u; i < 4u; ++i) {
            if (!isHex(input_[position_ + i])) return false;
            codePoint = (codePoint << 4u) | hexValue(input_[position_ + i]);
          }
          position_ += 4u;
          // Reject UTF-16 surrogate halves.  Canonical output is UTF-8 and
          // accepting an unpaired half would make string equality ambiguous.
          if (!appendUtf8(out, codePoint)) return false;
          break;
        }
        default: return false;
      }
    }
    return false;
  }

  bool parseNumber(std::string* out) {
    const std::size_t start = position_;
    if (position_ < input_.size() && input_[position_] == '-') ++position_;
    if (position_ >= input_.size()) return false;
    if (input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') return false;
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') return false;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction) return false;
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent) return false;
    }
    *out = std::string(input_.substr(start, position_ - start));
    return !out->empty();
  }

  bool parseObject(JsonValue* value, std::size_t depth) {
    if (input_[position_] != '{') return false;
    ++position_;
    value->type = JsonValue::Type::Object;
    value->object.clear();
    skipWhitespace();
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_;
      return true;
    }
    for (std::size_t count = 0u; count < kMaxJsonMembers; ++count) {
      skipWhitespace();
      std::string key;
      if (!parseString(&key)) return false;
      for (const auto& item : value->object) {
        if (item.first == key) return false;
      }
      skipWhitespace();
      if (position_ >= input_.size() || input_[position_] != ':') return false;
      ++position_;
      skipWhitespace();
      JsonValue child;
      if (!parseValue(&child, depth)) return false;
      value->object.emplace_back(std::move(key), std::move(child));
      skipWhitespace();
      if (position_ >= input_.size()) return false;
      if (input_[position_] == '}') {
        ++position_;
        return true;
      }
      if (input_[position_] != ',') return false;
      ++position_;
    }
    return false;
  }

  bool parseArray(JsonValue* value, std::size_t depth) {
    if (input_[position_] != '[') return false;
    ++position_;
    value->type = JsonValue::Type::Array;
    value->array.clear();
    skipWhitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_;
      return true;
    }
    for (std::size_t count = 0u; count < kMaxJsonArrayItems; ++count) {
      skipWhitespace();
      JsonValue child;
      if (!parseValue(&child, depth)) return false;
      value->array.emplace_back(std::move(child));
      skipWhitespace();
      if (position_ >= input_.size()) return false;
      if (input_[position_] == ']') {
        ++position_;
        return true;
      }
      if (input_[position_] != ',') return false;
      ++position_;
    }
    return false;
  }

  void skipWhitespace() noexcept {
    while (position_ < input_.size()) {
      const char c = input_[position_];
      if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
      ++position_;
    }
  }

  std::string_view input_;
  std::size_t position_ = 0u;
};

bool parseJson(std::string_view text, JsonValue* output) {
  if (!output || text.empty() || text.size() > kViewerWorkspacePersistenceMaxDocumentBytes) {
    return false;
  }
  JsonParser parser(text);
  return parser.parse(output);
}

const JsonValue* member(const JsonValue& object, std::string_view name) noexcept {
  if (object.type != JsonValue::Type::Object) return nullptr;
  for (const auto& item : object.object) {
    if (item.first == name) return &item.second;
  }
  return nullptr;
}

bool objectHasOnly(const JsonValue& object,
                  std::initializer_list<std::string_view> allowed) noexcept {
  if (object.type != JsonValue::Type::Object) return false;
  for (const auto& item : object.object) {
    bool found = false;
    for (const std::string_view key : allowed) {
      if (item.first == key) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

bool runtimeKey(std::string_view key) noexcept {
#define RUNTIME_KEY(name) if (key == #name) return true
  RUNTIME_KEY(stateRevision);
  RUNTIME_KEY(sampleSettingsKey);
  RUNTIME_KEY(plotModel);
  RUNTIME_KEY(circularHsl);
  RUNTIME_KEY(circularHsv);
  RUNTIME_KEY(normConeNormalized);
  RUNTIME_KEY(plotDisplayLinear);
  RUNTIME_KEY(plotDisplayLinearTransfer);
  RUNTIME_KEY(liveUpdate);
  RUNTIME_KEY(updateMode);
  RUNTIME_KEY(quality);
  RUNTIME_KEY(scale);
  RUNTIME_KEY(sampling);
  RUNTIME_KEY(occupancyGuidedFill);
  RUNTIME_KEY(plotStyle);
  RUNTIME_KEY(pointShape);
  RUNTIME_KEY(pointSize);
  RUNTIME_KEY(colorSaturation);
  RUNTIME_KEY(keepOnTop);
  RUNTIME_KEY(resetViewOnPlotSwitch);
  RUNTIME_KEY(showOverflow);
  RUNTIME_KEY(highlightOverflow);
  RUNTIME_KEY(overflowHighlightR);
  RUNTIME_KEY(overflowHighlightG);
  RUNTIME_KEY(overflowHighlightB);
  RUNTIME_KEY(backgroundR);
  RUNTIME_KEY(backgroundG);
  RUNTIME_KEY(backgroundB);
  RUNTIME_KEY(fillVolume);
  RUNTIME_KEY(fillGrayRamp);
  RUNTIME_KEY(fillResolution);
  RUNTIME_KEY(readGrayRamp);
  RUNTIME_KEY(readIdentityPlot);
  RUNTIME_KEY(isolateIdentityData);
  RUNTIME_KEY(excludeIdentityData);
  RUNTIME_KEY(identityReadResolution);
  RUNTIME_KEY(volumeSliceLassoRegion);
  RUNTIME_KEY(volumeSliceRed);
  RUNTIME_KEY(volumeSliceYellow);
  RUNTIME_KEY(volumeSliceGreen);
  RUNTIME_KEY(volumeSliceCyan);
  RUNTIME_KEY(volumeSliceBlue);
  RUNTIME_KEY(volumeSliceMagenta);
  RUNTIME_KEY(neutralRadius);
  RUNTIME_KEY(chromaticityInputPrimaries);
  RUNTIME_KEY(chromaticityInputTransfer);
  RUNTIME_KEY(chromaticityReferenceBasis);
  RUNTIME_KEY(chromaticityOverlayPrimaries);
  RUNTIME_KEY(chromaticityPlanckianLocus);
  RUNTIME_KEY(chromaticitySpectralLocus3D);
  RUNTIME_KEY(glossNeighborhood);
  RUNTIME_KEY(glossLiftScale);
  RUNTIME_KEY(glossSpatialInset);
  RUNTIME_KEY(glossBodyOpacity);
  RUNTIME_KEY(glossHighlightOpacity);
  RUNTIME_KEY(glossPointCrispness);
  RUNTIME_KEY(glossHideText);
  RUNTIME_KEY(waveformMode);
  RUNTIME_KEY(waveformHighDetail);
  RUNTIME_KEY(waveformContinuousHighDetail);
  RUNTIME_KEY(waveformHighDetailRequested);
  RUNTIME_KEY(waveformSampleColumns);
  RUNTIME_KEY(waveformSamplesPerColumn);
  RUNTIME_KEY(waveformPointBrightness);
  RUNTIME_KEY(waveformGridBrightness);
  RUNTIME_KEY(waveformSaturation);
  RUNTIME_KEY(waveformDotSize);
  RUNTIME_KEY(waveformChannelRed);
  RUNTIME_KEY(waveformChannelGreen);
  RUNTIME_KEY(waveformChannelBlue);
  RUNTIME_KEY(waveformChannelLuma);
  RUNTIME_KEY(waveformShowOverflow);
  RUNTIME_KEY(waveformHighlightOverflow);
  RUNTIME_KEY(waveformLumaMethod);
  RUNTIME_KEY(histogramMode);
  RUNTIME_KEY(histogramShowOverflow);
  RUNTIME_KEY(histogramHighlightOverflow);
  RUNTIME_KEY(scopeRangeMode);
  RUNTIME_KEY(sourceDetailMode);
  RUNTIME_KEY(sourceMaxProxyLongEdge);
  RUNTIME_KEY(sourceUseNativeWhenAvailable);
  RUNTIME_KEY(sourceSyncSelections);
  RUNTIME_KEY(sourceSyncCommonPlotSettings);
#undef RUNTIME_KEY
  return false;
}

bool requiredMember(const JsonValue& object, std::string_view name,
                    const JsonValue** value) noexcept {
  if (!value) return false;
  *value = member(object, name);
  return *value != nullptr;
}

bool readBool(const JsonValue& object, std::string_view name, bool* output,
              bool required = true) noexcept {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (value->type != JsonValue::Type::Bool || !output) return false;
  *output = value->boolean;
  return true;
}

bool parseInt64(std::string_view text, int64_t* output) {
  if (!output || text.empty()) return false;
  std::string copy(text);
  char* end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll(copy.c_str(), &end, 10);
  if (errno == ERANGE || end != copy.c_str() + copy.size()) return false;
  *output = static_cast<int64_t>(parsed);
  return true;
}

bool parseUInt64(std::string_view text, uint64_t* output) {
  if (!output || text.empty() || text.front() == '-') return false;
  std::string copy(text);
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(copy.c_str(), &end, 10);
  if (errno == ERANGE || end != copy.c_str() + copy.size()) return false;
  *output = static_cast<uint64_t>(parsed);
  return true;
}

bool parseDouble(std::string_view text, double* output) {
  if (!output || text.empty()) return false;
  std::string copy(text);
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(copy.c_str(), &end);
  if (errno == ERANGE || end != copy.c_str() + copy.size() || !finite(parsed)) {
    return false;
  }
  *output = parsed;
  return true;
}

bool readInt(const JsonValue& object, std::string_view name, int* output,
             int minimum, int maximum, bool required = true) {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (!output || value->type != JsonValue::Type::Number) return false;
  int64_t parsed = 0;
  if (!parseInt64(value->number, &parsed) || parsed < minimum || parsed > maximum) {
    return false;
  }
  *output = static_cast<int>(parsed);
  return true;
}

bool readUInt64(const JsonValue& object, std::string_view name, uint64_t* output,
                uint64_t minimum, uint64_t maximum,
                bool required = true) {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (!output || value->type != JsonValue::Type::Number) return false;
  uint64_t parsed = 0u;
  if (!parseUInt64(value->number, &parsed) || parsed < minimum || parsed > maximum) {
    return false;
  }
  *output = parsed;
  return true;
}

bool readFloat(const JsonValue& object, std::string_view name, float* output,
               float minimum, float maximum, bool required = true) {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (!output || value->type != JsonValue::Type::Number) return false;
  double parsed = 0.0;
  if (!parseDouble(value->number, &parsed) || parsed < minimum || parsed > maximum) {
    return false;
  }
  *output = static_cast<float>(parsed);
  return finite(*output);
}

bool readDouble(const JsonValue& object, std::string_view name, double* output,
                double minimum, double maximum, bool required = true) {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (!output || value->type != JsonValue::Type::Number ||
      !parseDouble(value->number, output) || *output < minimum || *output > maximum) {
    return false;
  }
  return true;
}

bool readString(const JsonValue& object, std::string_view name, std::string* output,
                std::size_t maximum, bool required = true,
                bool allowEmpty = true) {
  const JsonValue* value = member(object, name);
  if (!value) return !required;
  if (!output || value->type != JsonValue::Type::String ||
      value->string.size() > maximum || (!allowEmpty && value->string.empty())) {
    return false;
  }
  *output = value->string;
  return true;
}

bool finiteRect(const PlotWindowRectNorm& rect) noexcept {
  return finite(rect.x) && finite(rect.y) && finite(rect.w) && finite(rect.h) &&
         rect.x >= -kRectEpsilon && rect.y >= -kRectEpsilon && rect.w > 0.0f &&
         rect.h > 0.0f && rect.x + rect.w <= 1.0f + kRectEpsilon &&
         rect.y + rect.h <= 1.0f + kRectEpsilon;
}

bool finiteCamera(const CameraState& camera) noexcept {
  return finite(camera.qx) && finite(camera.qy) && finite(camera.qz) &&
         finite(camera.qw) && finite(camera.distance) && finite(camera.panX) &&
         finite(camera.panY) && camera.distance > 0.0f &&
         camera.orthographicView >= -1 && camera.orthographicView <= 7;
}

std::string escapeJson(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8u);
  for (unsigned char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20u) {
          std::ostringstream hex;
          hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c);
          out += hex.str();
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

template <typename Number>
std::string numberString(Number value, int precision) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(precision) << std::defaultfloat << value;
  return stream.str();
}

void appendKey(std::string* output, std::string_view key) {
  if (!output) return;
  if (output->size() > 1u && output->back() != '{' && output->back() != '[') {
    output->push_back(',');
  }
  output->push_back('"');
  output->append(key.data(), key.size());
  output->append("\":");
}

void appendString(std::string* output, std::string_view key, std::string_view value) {
  appendKey(output, key);
  output->push_back('"');
  output->append(escapeJson(value));
  output->push_back('"');
}

void appendBool(std::string* output, std::string_view key, bool value) {
  appendKey(output, key);
  output->append(value ? "true" : "false");
}

void appendInt(std::string* output, std::string_view key, int value) {
  appendKey(output, key);
  output->append(numberString(value, 32));
}

void appendUInt64(std::string* output, std::string_view key, uint64_t value) {
  appendKey(output, key);
  output->append(numberString(value, 32));
}

void appendFloat(std::string* output, std::string_view key, float value) {
  appendKey(output, key);
  output->append(numberString(value, 9));
}

void appendDouble(std::string* output, std::string_view key, double value) {
  appendKey(output, key);
  output->append(numberString(value, 17));
}

void beginObject(std::string* output) { output->push_back('{'); }
void endObject(std::string* output) { output->push_back('}'); }

uint64_t fnv1a(std::string_view bytes) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : bytes) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string checksumString(uint64_t checksum) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::hex << std::setw(16) << std::setfill('0') << checksum;
  return stream.str();
}

bool parseChecksum(std::string_view text, uint64_t* output) noexcept {
  if (!output || text.size() != 16u) return false;
  uint64_t result = 0u;
  for (char c : text) {
    if (!isHex(c)) return false;
    result = (result << 4u) | hexValue(c);
  }
  *output = result;
  return true;
}

void appendRect(std::string* output, const PlotWindowRectNorm& rect) {
  beginObject(output);
  appendFloat(output, "x", rect.x);
  appendFloat(output, "y", rect.y);
  appendFloat(output, "w", rect.w);
  appendFloat(output, "h", rect.h);
  endObject(output);
}

bool parseRect(const JsonValue& value, PlotWindowRectNorm* rect, bool clamp) {
  if (!rect || value.type != JsonValue::Type::Object) return false;
  if (!clamp && !objectHasOnly(value, {"x", "y", "w", "h"})) return false;
  PlotWindowRectNorm parsed{};
  const float min = clamp ? -std::numeric_limits<float>::max() : -1.0f;
  if (!readFloat(value, "x", &parsed.x, min, std::numeric_limits<float>::max()) ||
      !readFloat(value, "y", &parsed.y, min, std::numeric_limits<float>::max()) ||
      !readFloat(value, "w", &parsed.w, -std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()) ||
      !readFloat(value, "h", &parsed.h, -std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max())) {
    return false;
  }
  if (clamp) {
    parsed.x = std::max(0.0f, std::min(1.0f, parsed.x));
    parsed.y = std::max(0.0f, std::min(1.0f, parsed.y));
    parsed.w = std::max(0.05f, std::min(1.0f, parsed.w));
    parsed.h = std::max(0.05f, std::min(1.0f, parsed.h));
    if (parsed.x + parsed.w > 1.0f) parsed.x = std::max(0.0f, 1.0f - parsed.w);
    if (parsed.y + parsed.h > 1.0f) parsed.y = std::max(0.0f, 1.0f - parsed.h);
  }
  if (!finiteRect(parsed)) return false;
  *rect = parsed;
  return true;
}

void appendCamera(std::string* output, const CameraState& camera) {
  beginObject(output);
  appendFloat(output, "qx", camera.qx);
  appendFloat(output, "qy", camera.qy);
  appendFloat(output, "qz", camera.qz);
  appendFloat(output, "qw", camera.qw);
  appendFloat(output, "distance", camera.distance);
  appendFloat(output, "panX", camera.panX);
  appendFloat(output, "panY", camera.panY);
  appendBool(output, "orthographic", camera.orthographic);
  appendInt(output, "orthographicView", camera.orthographicView);
  endObject(output);
}

bool parseCamera(const JsonValue& value, CameraState* camera, bool strict) {
  if (!camera || value.type != JsonValue::Type::Object) return false;
  if (strict && !objectHasOnly(value, {"qx", "qy", "qz", "qw", "distance",
                                       "panX", "panY", "orthographic",
                                       "orthographicView"})) {
    return false;
  }
  CameraState parsed{};
  const float lo = -std::numeric_limits<float>::max();
  if (!readFloat(value, "qx", &parsed.qx, lo, std::numeric_limits<float>::max()) ||
      !readFloat(value, "qy", &parsed.qy, lo, std::numeric_limits<float>::max()) ||
      !readFloat(value, "qz", &parsed.qz, lo, std::numeric_limits<float>::max()) ||
      !readFloat(value, "qw", &parsed.qw, lo, std::numeric_limits<float>::max()) ||
      !readFloat(value, "distance", &parsed.distance, 0.000001f,
                 std::numeric_limits<float>::max()) ||
      !readFloat(value, "panX", &parsed.panX, lo, std::numeric_limits<float>::max()) ||
      !readFloat(value, "panY", &parsed.panY, lo, std::numeric_limits<float>::max()) ||
      !readBool(value, "orthographic", &parsed.orthographic) ||
      !readInt(value, "orthographicView", &parsed.orthographicView, -1, 7)) {
    return false;
  }
  if (strict && !finiteCamera(parsed)) return false;
  *camera = parsed;
  return true;
}

void recomputeBounds(LassoStroke* stroke) noexcept {
  if (!stroke || stroke->points.empty()) {
    if (stroke) stroke->boundsValid = false;
    return;
  }
  stroke->minXNorm = stroke->maxXNorm = stroke->points.front().xNorm;
  stroke->minYNorm = stroke->maxYNorm = stroke->points.front().yNorm;
  for (const auto& point : stroke->points) {
    stroke->minXNorm = std::min(stroke->minXNorm, point.xNorm);
    stroke->maxXNorm = std::max(stroke->maxXNorm, point.xNorm);
    stroke->minYNorm = std::min(stroke->minYNorm, point.yNorm);
    stroke->maxYNorm = std::max(stroke->maxYNorm, point.yNorm);
  }
  stroke->boundsValid = true;
}

// The viewer's lasso payload is a derived wire representation of the durable
// stroke list.  Keep the persistence codec in lockstep with the production
// serializer so a restored document cannot carry two conflicting selections.
std::string canonicalLassoData(uint64_t revision,
                               const std::vector<LassoStroke>& strokes) {
  std::string encoded;
  return encodeCanonicalViewerLassoData(revision, strokes, &encoded)
             ? encoded
             : std::string{};
}

bool validPersistedLassoStrokes(const std::vector<LassoStroke>& strokes) noexcept {
  if (strokes.size() > kViewerWorkspaceMaxLassoStrokes) return false;
  std::size_t totalPoints = 0u;
  for (const auto& stroke : strokes) {
    if (stroke.points.size() < 3u ||
        stroke.points.size() > kViewerWorkspaceMaxLassoPointsPerStroke ||
        stroke.points.size() > kViewerWorkspaceMaxLassoPoints - totalPoints) {
      return false;
    }
    totalPoints += stroke.points.size();
  }
  return true;
}

void appendLasso(std::string* output, const PlotWindowDomainState& window) {
  beginObject(output);
  appendUInt64(output, "revision", window.viewerLassoRevision);
  appendString(output, "data", window.viewerLassoData);
  appendKey(output, "strokes");
  output->push_back('[');
  for (std::size_t i = 0; i < window.viewerLassoStrokes.size(); ++i) {
    if (i) output->push_back(',');
    beginObject(output);
    appendBool(output, "subtract", window.viewerLassoStrokes[i].subtract);
    appendKey(output, "points");
    output->push_back('[');
    const auto& points = window.viewerLassoStrokes[i].points;
    for (std::size_t j = 0; j < points.size(); ++j) {
      if (j) output->push_back(',');
      beginObject(output);
      appendFloat(output, "x", points[j].xNorm);
      appendFloat(output, "y", points[j].yNorm);
      endObject(output);
    }
    output->push_back(']');
    endObject(output);
  }
  output->push_back(']');
  endObject(output);
}

bool parseLasso(const JsonValue& value, PlotWindowDomainState* window,
                bool strict) {
  if (!window || value.type != JsonValue::Type::Object) return false;
  if (strict && !objectHasOnly(value, {"revision", "data", "strokes"})) return false;
  uint64_t revision = 0u;
  std::string data;
  if (!readUInt64(value, "revision", &revision, 0u,
                  std::numeric_limits<uint64_t>::max()) ||
      !readString(value, "data", &data, kViewerWorkspacePersistenceMaxLassoBytes)) {
    return false;
  }
  const JsonValue* strokes = member(value, "strokes");
  if (!strokes || strokes->type != JsonValue::Type::Array ||
      strokes->array.size() > kViewerWorkspaceMaxLassoStrokes) {
    return false;
  }
  std::vector<LassoStroke> parsed;
  parsed.reserve(strokes->array.size());
  std::size_t totalPoints = 0u;
  for (const JsonValue& strokeValue : strokes->array) {
    if (strokeValue.type != JsonValue::Type::Object) return false;
    if (strict && !objectHasOnly(strokeValue, {"subtract", "points"})) return false;
    LassoStroke stroke{};
    if (!readBool(strokeValue, "subtract", &stroke.subtract)) return false;
    const JsonValue* points = member(strokeValue, "points");
    if (!points || points->type != JsonValue::Type::Array ||
        (strict && points->array.size() < 3u) ||
        points->array.size() > kViewerWorkspaceMaxLassoPointsPerStroke ||
        points->array.size() > kViewerWorkspaceMaxLassoPoints - totalPoints) {
      return false;
    }
    stroke.points.reserve(points->array.size());
    for (const JsonValue& pointValue : points->array) {
      if (pointValue.type != JsonValue::Type::Object) return false;
      if (strict && !objectHasOnly(pointValue, {"x", "y"})) return false;
      LassoPointNorm point{};
      if (!readFloat(pointValue, "x", &point.xNorm, 0.0f, 1.0f) ||
          !readFloat(pointValue, "y", &point.yNorm, 0.0f, 1.0f)) {
        return false;
      }
      stroke.points.push_back(point);
    }
    totalPoints += stroke.points.size();
    recomputeBounds(&stroke);
    parsed.push_back(std::move(stroke));
  }
  if (strict && !parsed.empty() && revision == 0u) return false;
  if (strict && data.size() > kViewerWorkspacePersistenceMaxLassoBytes) return false;
  if (strict && data != canonicalLassoData(revision, parsed)) return false;
  (void)strict;
  window->viewerLassoRevision = revision;
  window->viewerLassoData = std::move(data);
  window->viewerLassoStrokes = std::move(parsed);
  return true;
}

void appendRuntime(std::string* output, const ViewerRuntimeState& state) {
  beginObject(output);
  appendUInt64(output, "stateRevision", state.stateRevision);
  appendString(output, "sampleSettingsKey", state.sampleSettingsKey);
#define APPEND_RUNTIME_BOOL(name) appendBool(output, #name, state.name)
#define APPEND_RUNTIME_INT(name) appendInt(output, #name, state.name)
#define APPEND_RUNTIME_UINT(name) appendUInt64(output, #name, state.name)
#define APPEND_RUNTIME_DOUBLE(name) appendDouble(output, #name, state.name)
  APPEND_RUNTIME_INT(plotModel);
  APPEND_RUNTIME_BOOL(circularHsl);
  APPEND_RUNTIME_BOOL(circularHsv);
  APPEND_RUNTIME_BOOL(normConeNormalized);
  APPEND_RUNTIME_BOOL(plotDisplayLinear);
  APPEND_RUNTIME_INT(plotDisplayLinearTransfer);
  APPEND_RUNTIME_BOOL(liveUpdate);
  APPEND_RUNTIME_INT(updateMode);
  APPEND_RUNTIME_INT(quality);
  APPEND_RUNTIME_INT(scale);
  APPEND_RUNTIME_INT(sampling);
  APPEND_RUNTIME_BOOL(occupancyGuidedFill);
  APPEND_RUNTIME_INT(plotStyle);
  APPEND_RUNTIME_INT(pointShape);
  APPEND_RUNTIME_DOUBLE(pointSize);
  APPEND_RUNTIME_DOUBLE(colorSaturation);
  APPEND_RUNTIME_BOOL(keepOnTop);
  APPEND_RUNTIME_BOOL(resetViewOnPlotSwitch);
  APPEND_RUNTIME_BOOL(showOverflow);
  APPEND_RUNTIME_BOOL(highlightOverflow);
  APPEND_RUNTIME_DOUBLE(overflowHighlightR);
  APPEND_RUNTIME_DOUBLE(overflowHighlightG);
  APPEND_RUNTIME_DOUBLE(overflowHighlightB);
  APPEND_RUNTIME_DOUBLE(backgroundR);
  APPEND_RUNTIME_DOUBLE(backgroundG);
  APPEND_RUNTIME_DOUBLE(backgroundB);
  APPEND_RUNTIME_BOOL(fillVolume);
  APPEND_RUNTIME_BOOL(fillGrayRamp);
  APPEND_RUNTIME_INT(fillResolution);
  APPEND_RUNTIME_BOOL(readGrayRamp);
  APPEND_RUNTIME_BOOL(readIdentityPlot);
  APPEND_RUNTIME_BOOL(isolateIdentityData);
  APPEND_RUNTIME_BOOL(excludeIdentityData);
  APPEND_RUNTIME_INT(identityReadResolution);
  APPEND_RUNTIME_BOOL(volumeSliceLassoRegion);
  APPEND_RUNTIME_BOOL(volumeSliceRed);
  APPEND_RUNTIME_BOOL(volumeSliceYellow);
  APPEND_RUNTIME_BOOL(volumeSliceGreen);
  APPEND_RUNTIME_BOOL(volumeSliceCyan);
  APPEND_RUNTIME_BOOL(volumeSliceBlue);
  APPEND_RUNTIME_BOOL(volumeSliceMagenta);
  APPEND_RUNTIME_DOUBLE(neutralRadius);
  APPEND_RUNTIME_INT(chromaticityInputPrimaries);
  APPEND_RUNTIME_INT(chromaticityInputTransfer);
  APPEND_RUNTIME_INT(chromaticityReferenceBasis);
  APPEND_RUNTIME_INT(chromaticityOverlayPrimaries);
  APPEND_RUNTIME_BOOL(chromaticityPlanckianLocus);
  APPEND_RUNTIME_BOOL(chromaticitySpectralLocus3D);
  APPEND_RUNTIME_INT(glossNeighborhood);
  APPEND_RUNTIME_DOUBLE(glossLiftScale);
  APPEND_RUNTIME_BOOL(glossSpatialInset);
  APPEND_RUNTIME_DOUBLE(glossBodyOpacity);
  APPEND_RUNTIME_DOUBLE(glossHighlightOpacity);
  APPEND_RUNTIME_DOUBLE(glossPointCrispness);
  APPEND_RUNTIME_BOOL(glossHideText);
  APPEND_RUNTIME_INT(waveformMode);
  APPEND_RUNTIME_BOOL(waveformHighDetail);
  APPEND_RUNTIME_BOOL(waveformContinuousHighDetail);
  APPEND_RUNTIME_BOOL(waveformHighDetailRequested);
  APPEND_RUNTIME_INT(waveformSampleColumns);
  APPEND_RUNTIME_INT(waveformSamplesPerColumn);
  APPEND_RUNTIME_DOUBLE(waveformPointBrightness);
  APPEND_RUNTIME_DOUBLE(waveformGridBrightness);
  APPEND_RUNTIME_DOUBLE(waveformSaturation);
  APPEND_RUNTIME_DOUBLE(waveformDotSize);
  APPEND_RUNTIME_BOOL(waveformChannelRed);
  APPEND_RUNTIME_BOOL(waveformChannelGreen);
  APPEND_RUNTIME_BOOL(waveformChannelBlue);
  APPEND_RUNTIME_BOOL(waveformChannelLuma);
  APPEND_RUNTIME_BOOL(waveformShowOverflow);
  APPEND_RUNTIME_BOOL(waveformHighlightOverflow);
  APPEND_RUNTIME_INT(waveformLumaMethod);
  APPEND_RUNTIME_INT(histogramMode);
  APPEND_RUNTIME_BOOL(histogramShowOverflow);
  APPEND_RUNTIME_BOOL(histogramHighlightOverflow);
  APPEND_RUNTIME_INT(scopeRangeMode);
  APPEND_RUNTIME_INT(sourceDetailMode);
  APPEND_RUNTIME_INT(sourceMaxProxyLongEdge);
  APPEND_RUNTIME_BOOL(sourceUseNativeWhenAvailable);
  APPEND_RUNTIME_BOOL(sourceSyncSelections);
  APPEND_RUNTIME_BOOL(sourceSyncCommonPlotSettings);
#undef APPEND_RUNTIME_BOOL
#undef APPEND_RUNTIME_INT
#undef APPEND_RUNTIME_UINT
#undef APPEND_RUNTIME_DOUBLE
  endObject(output);
}

bool runtimeDurableEqual(const ViewerRuntimeState& a,
                         const ViewerRuntimeState& b) noexcept {
  if (a.stateRevision != b.stateRevision ||
      a.sampleSettingsKey != b.sampleSettingsKey) {
    return false;
  }
#define CMP_RUNTIME(name) if (a.name != b.name) return false
  CMP_RUNTIME(plotModel);
  CMP_RUNTIME(circularHsl);
  CMP_RUNTIME(circularHsv);
  CMP_RUNTIME(normConeNormalized);
  CMP_RUNTIME(plotDisplayLinear);
  CMP_RUNTIME(plotDisplayLinearTransfer);
  CMP_RUNTIME(liveUpdate);
  CMP_RUNTIME(updateMode);
  CMP_RUNTIME(quality);
  CMP_RUNTIME(scale);
  CMP_RUNTIME(sampling);
  CMP_RUNTIME(occupancyGuidedFill);
  CMP_RUNTIME(plotStyle);
  CMP_RUNTIME(pointShape);
  CMP_RUNTIME(pointSize);
  CMP_RUNTIME(colorSaturation);
  CMP_RUNTIME(keepOnTop);
  CMP_RUNTIME(resetViewOnPlotSwitch);
  CMP_RUNTIME(showOverflow);
  CMP_RUNTIME(highlightOverflow);
  CMP_RUNTIME(overflowHighlightR);
  CMP_RUNTIME(overflowHighlightG);
  CMP_RUNTIME(overflowHighlightB);
  CMP_RUNTIME(backgroundR);
  CMP_RUNTIME(backgroundG);
  CMP_RUNTIME(backgroundB);
  CMP_RUNTIME(fillVolume);
  CMP_RUNTIME(fillGrayRamp);
  CMP_RUNTIME(fillResolution);
  CMP_RUNTIME(readGrayRamp);
  CMP_RUNTIME(readIdentityPlot);
  CMP_RUNTIME(isolateIdentityData);
  CMP_RUNTIME(excludeIdentityData);
  CMP_RUNTIME(identityReadResolution);
  CMP_RUNTIME(volumeSliceLassoRegion);
  CMP_RUNTIME(volumeSliceRed);
  CMP_RUNTIME(volumeSliceYellow);
  CMP_RUNTIME(volumeSliceGreen);
  CMP_RUNTIME(volumeSliceCyan);
  CMP_RUNTIME(volumeSliceBlue);
  CMP_RUNTIME(volumeSliceMagenta);
  CMP_RUNTIME(neutralRadius);
  CMP_RUNTIME(chromaticityInputPrimaries);
  CMP_RUNTIME(chromaticityInputTransfer);
  CMP_RUNTIME(chromaticityReferenceBasis);
  CMP_RUNTIME(chromaticityOverlayPrimaries);
  CMP_RUNTIME(chromaticityPlanckianLocus);
  CMP_RUNTIME(chromaticitySpectralLocus3D);
  CMP_RUNTIME(glossNeighborhood);
  CMP_RUNTIME(glossLiftScale);
  CMP_RUNTIME(glossSpatialInset);
  CMP_RUNTIME(glossBodyOpacity);
  CMP_RUNTIME(glossHighlightOpacity);
  CMP_RUNTIME(glossPointCrispness);
  CMP_RUNTIME(glossHideText);
  CMP_RUNTIME(waveformMode);
  CMP_RUNTIME(waveformHighDetail);
  CMP_RUNTIME(waveformContinuousHighDetail);
  CMP_RUNTIME(waveformHighDetailRequested);
  CMP_RUNTIME(waveformSampleColumns);
  CMP_RUNTIME(waveformSamplesPerColumn);
  CMP_RUNTIME(waveformPointBrightness);
  CMP_RUNTIME(waveformGridBrightness);
  CMP_RUNTIME(waveformSaturation);
  CMP_RUNTIME(waveformDotSize);
  CMP_RUNTIME(waveformChannelRed);
  CMP_RUNTIME(waveformChannelGreen);
  CMP_RUNTIME(waveformChannelBlue);
  CMP_RUNTIME(waveformChannelLuma);
  CMP_RUNTIME(waveformShowOverflow);
  CMP_RUNTIME(waveformHighlightOverflow);
  CMP_RUNTIME(waveformLumaMethod);
  CMP_RUNTIME(histogramMode);
  CMP_RUNTIME(histogramShowOverflow);
  CMP_RUNTIME(histogramHighlightOverflow);
  CMP_RUNTIME(scopeRangeMode);
  CMP_RUNTIME(sourceDetailMode);
  CMP_RUNTIME(sourceMaxProxyLongEdge);
  CMP_RUNTIME(sourceUseNativeWhenAvailable);
  CMP_RUNTIME(sourceSyncSelections);
  CMP_RUNTIME(sourceSyncCommonPlotSettings);
#undef CMP_RUNTIME
  return true;
}

bool parseRuntime(const JsonValue& value, ViewerRuntimeState* output,
                  bool strict) {
  if (!output || value.type != JsonValue::Type::Object) return false;
  if (strict) {
    for (const auto& item : value.object) {
      if (!runtimeKey(item.first)) {
        return false;
      }
    }
  }
  ViewerRuntimeState parsed{};
    if (!readUInt64(value, "stateRevision", &parsed.stateRevision, 1u,
                    std::numeric_limits<uint64_t>::max()) ||
        !readString(value, "sampleSettingsKey", &parsed.sampleSettingsKey,
                    kViewerWorkspacePersistenceMaxRuntimeStringBytes)) {
      return false;
    }
#define READ_BOOL(name) if (!readBool(value, #name, &parsed.name)) return false
#define READ_INT(name, lo, hi) if (!readInt(value, #name, &parsed.name, lo, hi)) return false
#define READ_DOUBLE(name, lo, hi) if (!readDouble(value, #name, &parsed.name, lo, hi)) return false
    READ_INT(plotModel, 0, kPlotModelCount - 1);
    READ_BOOL(circularHsl);
    READ_BOOL(circularHsv);
    READ_BOOL(normConeNormalized);
    READ_BOOL(plotDisplayLinear);
    READ_INT(plotDisplayLinearTransfer, 0, 4096);
    READ_BOOL(liveUpdate);
    READ_INT(updateMode, 0, 2);
    READ_INT(quality, 0, 2);
    READ_INT(scale, 0, 3);
    READ_INT(sampling, 0, 2);
    READ_BOOL(occupancyGuidedFill);
    READ_INT(plotStyle, 0, 1);
    READ_INT(pointShape, 0, 1);
    READ_DOUBLE(pointSize, 0.35, 3.0);
    READ_DOUBLE(colorSaturation, 0.8, 6.0);
    READ_BOOL(keepOnTop);
    READ_BOOL(resetViewOnPlotSwitch);
    READ_BOOL(showOverflow);
    READ_BOOL(highlightOverflow);
    READ_DOUBLE(overflowHighlightR, 0.0, 1.0);
    READ_DOUBLE(overflowHighlightG, 0.0, 1.0);
    READ_DOUBLE(overflowHighlightB, 0.0, 1.0);
    READ_DOUBLE(backgroundR, 0.0, 1.0);
    READ_DOUBLE(backgroundG, 0.0, 1.0);
    READ_DOUBLE(backgroundB, 0.0, 1.0);
    READ_BOOL(fillVolume);
    READ_BOOL(fillGrayRamp);
    READ_INT(fillResolution, 4, 65);
    READ_BOOL(readGrayRamp);
    READ_BOOL(readIdentityPlot);
    READ_BOOL(isolateIdentityData);
    READ_BOOL(excludeIdentityData);
    READ_INT(identityReadResolution, 4, 65);
    READ_BOOL(volumeSliceLassoRegion);
    READ_BOOL(volumeSliceRed);
    READ_BOOL(volumeSliceYellow);
    READ_BOOL(volumeSliceGreen);
    READ_BOOL(volumeSliceCyan);
    READ_BOOL(volumeSliceBlue);
    READ_BOOL(volumeSliceMagenta);
    READ_DOUBLE(neutralRadius, 0.0, 1.0);
    READ_INT(chromaticityInputPrimaries, 0, 4096);
    READ_INT(chromaticityInputTransfer, 0, 4096);
    READ_INT(chromaticityReferenceBasis, 0, 1);
    READ_INT(chromaticityOverlayPrimaries, 0, 4096);
    READ_BOOL(chromaticityPlanckianLocus);
    READ_BOOL(chromaticitySpectralLocus3D);
    READ_INT(glossNeighborhood, 0, 2);
    READ_DOUBLE(glossLiftScale, 0.25, 3.0);
    READ_BOOL(glossSpatialInset);
    READ_DOUBLE(glossBodyOpacity, 0.0, 1.0);
    READ_DOUBLE(glossHighlightOpacity, 0.0, 1.0);
    READ_DOUBLE(glossPointCrispness, 0.0, 1.0);
    READ_BOOL(glossHideText);
    READ_INT(waveformMode, 0, 2);
    READ_BOOL(waveformHighDetail);
    READ_BOOL(waveformContinuousHighDetail);
    READ_BOOL(waveformHighDetailRequested);
    READ_INT(waveformSampleColumns, 0, 1536);
    READ_INT(waveformSamplesPerColumn, 0, 192);
    READ_DOUBLE(waveformPointBrightness, 0.1, 2.0);
    READ_DOUBLE(waveformGridBrightness, 0.0, 2.0);
    READ_DOUBLE(waveformSaturation, 0.0, 1.0);
    READ_DOUBLE(waveformDotSize, 0.05, 1.5);
    READ_BOOL(waveformChannelRed);
    READ_BOOL(waveformChannelGreen);
    READ_BOOL(waveformChannelBlue);
    READ_BOOL(waveformChannelLuma);
    READ_BOOL(waveformShowOverflow);
    READ_BOOL(waveformHighlightOverflow);
    READ_INT(waveformLumaMethod, 0, 3);
    READ_INT(histogramMode, 0, 1);
    READ_BOOL(histogramShowOverflow);
    READ_BOOL(histogramHighlightOverflow);
    READ_INT(scopeRangeMode, 0, 2);
    READ_INT(sourceDetailMode, 0, 4);
    READ_INT(sourceMaxProxyLongEdge, 768, 4096);
    READ_BOOL(sourceUseNativeWhenAvailable);
    READ_BOOL(sourceSyncSelections);
    READ_BOOL(sourceSyncCommonPlotSettings);
#undef READ_BOOL
#undef READ_INT
#undef READ_DOUBLE
    // These fields intentionally never cross the persistence boundary.
    parsed.refreshPolicy = "none";
    parsed.requiresHostSamples = false;
    parsed.sourceSessionId.clear();
    parsed.hostRefreshRequestedRevision = 0u;
    const ViewerRuntimeState clamped = clampedViewerRuntimeState(parsed);
    if (strict && !runtimeDurableEqual(parsed, clamped)) return false;
    *output = strict ? parsed : clamped;
  return true;
}

void appendWindow(std::string* output, const PlotWindowDomainState& window) {
  beginObject(output);
  appendString(output, "type", "window");
  appendInt(output, "windowId", window.windowId);
  appendKey(output, "rect");
  appendRect(output, window.rect);
  appendKey(output, "camera");
  appendCamera(output, window.camera);
  appendKey(output, "runtime");
  appendRuntime(output, window.viewState);
  appendKey(output, "lasso");
  appendLasso(output, window);
  appendBool(output, "slicingDrawerOpen", window.slicingDrawerOpen);
  endObject(output);
}

bool parseWindow(const JsonValue& value, PlotWindowDomainState* output,
                 bool strict) {
  if (!output || value.type != JsonValue::Type::Object) return false;
  if (strict && !objectHasOnly(value, {"type", "windowId", "rect", "camera",
                                       "runtime", "lasso", "slicingDrawerOpen"})) {
    return false;
  }
  PlotWindowDomainState parsed{};
  if (!readInt(value, "windowId", &parsed.windowId, 1,
               std::numeric_limits<int>::max()) ||
      !readBool(value, "slicingDrawerOpen", &parsed.slicingDrawerOpen)) {
    return false;
  }
  const JsonValue* child = nullptr;
  if (!requiredMember(value, "rect", &child) ||
      !parseRect(*child, &parsed.rect, !strict) ||
      !requiredMember(value, "camera", &child) ||
      !parseCamera(*child, &parsed.camera, strict) ||
      !requiredMember(value, "runtime", &child) ||
      !parseRuntime(*child, &parsed.viewState, strict) ||
      !requiredMember(value, "lasso", &child) ||
      !parseLasso(*child, &parsed, strict)) {
    return false;
  }
  parsed.syncLabel = "Waiting for Resolve";
  parsed.stableSyncLabel = "Waiting for Resolve";
  parsed.lastHealthySyncLabelTime = -10.0;
  parsed.fitRequested = false;
  parsed.selected = false;
  parsed.sourceSignalDocked = false;
  parsed.sourceSignalTemporaryLassoSurface = false;
  parsed.sourceSignalDockOwnerWindowId = -1;
  parsed.sourceSignalRestoreRect = {};
  parsed.sourceSignalDockAnimStart = -10.0;
  parsed.sourceSignalDockAnimatingToDock = false;
  parsed.slicingDrawerAnimStart = -10.0;
  *output = std::move(parsed);
  return true;
}

void appendDocumentRecord(std::string* output, const ViewerWorkspaceState& state) {
  beginObject(output);
  appendString(output, "type", "document");
  appendInt(output, "focusedWindowId", state.focusedWindowId);
  appendInt(output, "nextWindowId", state.nextWindowId);
  appendString(output, "layoutPresetSelection", state.layoutPresetSelection);
  appendString(output, "layoutPresetBeforeSolo", state.layoutPresetBeforeSolo);
  appendString(output, "layoutPresetNameInput", state.layoutPresetNameInput);
  appendUInt64(output, "revision", state.revision);
  appendBool(output, "sourceLassoSelectionsSynced", state.sourceLassoSelectionsSynced);
  appendInt(output, "sourceLassoTargetWindowId", state.sourceLassoTargetWindowId);
  appendBool(output, "sourceLassoHasSelection", state.sourceLassoHasSelection);
  appendBool(output, "sourceLassoGlobalHasSelection", state.sourceLassoGlobalHasSelection);
  PlotWindowDomainState sourceLasso{};
  sourceLasso.viewerLassoRevision = state.sourceLassoRevision;
  sourceLasso.viewerLassoStrokes = state.sourceLassoStrokes;
  sourceLasso.viewerLassoData = canonicalLassoData(
      sourceLasso.viewerLassoRevision, sourceLasso.viewerLassoStrokes);
  appendKey(output, "sourceLasso");
  appendLasso(output, sourceLasso);
  endObject(output);
}

void appendPresentationRecord(std::string* output,
                              const ViewerWorkspacePresentationPreferences& prefs) {
  beginObject(output);
  appendString(output, "type", "presentation");
  appendBool(output, "showWorkspaceButtons", prefs.showWorkspaceButtons);
  appendBool(output, "showSliceButtonInPlotWindows", prefs.showSliceButtonInPlotWindows);
  appendInt(output, "viewerFontSize", prefs.viewerFontSize);
  appendInt(output, "windowWidth", prefs.windowWidth);
  appendInt(output, "windowHeight", prefs.windowHeight);
  appendInt(output, "windowPosX", prefs.windowPosX);
  appendInt(output, "windowPosY", prefs.windowPosY);
  appendBool(output, "windowPositionValid", prefs.windowPositionValid);
  appendInt(output, "activeStandardLayoutIndex", prefs.activeStandardLayoutIndex);
  appendFloat(output, "workspaceTopNorm", prefs.workspaceTopNorm);
  endObject(output);
}

bool parseDocumentRecord(const JsonValue& value, ViewerWorkspaceState* state) {
  if (!state || value.type != JsonValue::Type::Object) return false;
  if (!objectHasOnly(value, {"type", "focusedWindowId", "nextWindowId",
                             "layoutPresetSelection", "layoutPresetBeforeSolo",
                             "layoutPresetNameInput", "revision",
                             "sourceLassoSelectionsSynced", "sourceLassoTargetWindowId",
                             "sourceLassoHasSelection", "sourceLassoGlobalHasSelection",
                             "sourceLasso"})) {
    return false;
  }
  if (!readInt(value, "focusedWindowId", &state->focusedWindowId, 1,
               std::numeric_limits<int>::max()) ||
      !readInt(value, "nextWindowId", &state->nextWindowId, 1,
               std::numeric_limits<int>::max()) ||
      !readString(value, "layoutPresetSelection", &state->layoutPresetSelection,
                   kViewerWorkspaceMaxStringBytes, true, false) ||
      !readString(value, "layoutPresetBeforeSolo", &state->layoutPresetBeforeSolo,
                   kViewerWorkspaceMaxStringBytes) ||
      !readString(value, "layoutPresetNameInput", &state->layoutPresetNameInput,
                   kViewerWorkspaceMaxStringBytes) ||
      !readUInt64(value, "revision", &state->revision, 1u,
                  std::numeric_limits<uint64_t>::max()) ||
      !readBool(value, "sourceLassoSelectionsSynced",
                &state->sourceLassoSelectionsSynced) ||
      !readInt(value, "sourceLassoTargetWindowId", &state->sourceLassoTargetWindowId,
               -1, std::numeric_limits<int>::max()) ||
      !readBool(value, "sourceLassoHasSelection", &state->sourceLassoHasSelection) ||
      !readBool(value, "sourceLassoGlobalHasSelection",
                &state->sourceLassoGlobalHasSelection)) {
    return false;
  }
  const JsonValue* sourceLasso = nullptr;
  if (!requiredMember(value, "sourceLasso", &sourceLasso)) return false;
  PlotWindowDomainState parsedLasso{};
  if (!parseLasso(*sourceLasso, &parsedLasso, true)) return false;
  state->sourceLassoRevision = parsedLasso.viewerLassoRevision;
  state->sourceLassoStrokes = std::move(parsedLasso.viewerLassoStrokes);
  const bool hasSelection = !state->sourceLassoStrokes.empty();
  if (state->sourceLassoGlobalHasSelection != hasSelection) {
    return false;
  }
  return true;
}

bool parsePresentationRecord(const JsonValue& value,
                             ViewerWorkspacePresentationPreferences* prefs,
                             bool strict) {
  if (!prefs || value.type != JsonValue::Type::Object) return false;
  if (strict && !objectHasOnly(value, {"type", "showWorkspaceButtons",
                                       "showSliceButtonInPlotWindows", "viewerFontSize",
                                       "windowWidth", "windowHeight", "windowPosX",
                                       "windowPosY", "windowPositionValid",
                                       "activeStandardLayoutIndex", "workspaceTopNorm"})) {
    return false;
  }
  if (!readBool(value, "showWorkspaceButtons", &prefs->showWorkspaceButtons) ||
      !readBool(value, "showSliceButtonInPlotWindows",
                &prefs->showSliceButtonInPlotWindows) ||
      !readInt(value, "viewerFontSize", &prefs->viewerFontSize, 0, 2) ||
      !readInt(value, "windowWidth", &prefs->windowWidth, 0, 16384) ||
      !readInt(value, "windowHeight", &prefs->windowHeight, 0, 16384) ||
      !readInt(value, "windowPosX", &prefs->windowPosX,
               std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) ||
      !readInt(value, "windowPosY", &prefs->windowPosY,
               std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) ||
      !readBool(value, "windowPositionValid", &prefs->windowPositionValid) ||
      !readInt(value, "activeStandardLayoutIndex", &prefs->activeStandardLayoutIndex,
               0, kViewerLayoutChoiceCount - 1) ||
      !readFloat(value, "workspaceTopNorm", &prefs->workspaceTopNorm, 0.0f, 1.0f)) {
    return false;
  }
  if (strict && (prefs->windowWidth < 0 || prefs->windowHeight < 0 ||
                 !finite(prefs->workspaceTopNorm))) {
    return false;
  }
  return true;
}

enum class SanitiseResult : uint8_t { Accepted, Invalid, AllocationFailure };
SanitiseResult sanitiseDocumentInternal(const ViewerWorkspaceDocument& input,
                                        ViewerWorkspaceDocument* output);

bool parseV2(std::string_view bytes, ViewerWorkspaceDocument* output,
             ViewerWorkspacePersistenceStatus* status) {
  if (!output || !status) return false;
  const std::size_t headerEnd = bytes.find('\n');
  if (headerEnd == std::string_view::npos || headerEnd == 0u ||
      bytes[headerEnd - 1u] == '\r') {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  JsonValue header;
  if (!parseJson(bytes.substr(0u, headerEnd), &header) ||
      header.type != JsonValue::Type::Object) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  if (!objectHasOnly(header, {"schema", "version", "payloadBytes", "checksum"})) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  std::string schema;
  uint64_t version = 0u;
  uint64_t payloadBytes = 0u;
  std::string checksum;
  if (!readString(header, "schema", &schema, 128u, true, false) ||
      !readUInt64(header, "version", &version, 0u,
                  std::numeric_limits<uint32_t>::max()) ||
      !readUInt64(header, "payloadBytes", &payloadBytes, 0u,
                  kViewerWorkspacePersistenceMaxDocumentBytes) ||
      !readString(header, "checksum", &checksum, 32u, true, false)) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  if (schema != kViewerWorkspacePersistenceSchema) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  if (version != kViewerWorkspacePersistenceVersion) {
    *status = ViewerWorkspacePersistenceStatus::UnsupportedVersion;
    return false;
  }
  const std::string_view payload = bytes.substr(headerEnd + 1u);
  uint64_t expectedChecksum = 0u;
  if (payloadBytes != payload.size() ||
      !parseChecksum(checksum, &expectedChecksum) ||
      fnv1a(payload) != expectedChecksum) {
    *status = ViewerWorkspacePersistenceStatus::IntegrityMismatch;
    return false;
  }
  if (payload.empty() || payload.back() != '\n') {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  ViewerWorkspaceDocument parsed = defaultViewerWorkspaceDocument();
  parsed.workspace.windows.clear();
  bool documentSeen = false;
  bool presentationSeen = false;
  std::size_t lineStart = 0u;
  std::size_t lineCount = 0u;
  while (lineStart < payload.size()) {
    const std::size_t newline = payload.find('\n', lineStart);
    if (newline == std::string_view::npos || newline == lineStart ||
        ++lineCount > kViewerWorkspacePersistenceMaxLines) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    std::string_view line = payload.substr(lineStart, newline - lineStart);
    if (!line.empty() && line.back() == '\r') {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    if (line.empty()) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    JsonValue record;
    if (!parseJson(line, &record) || record.type != JsonValue::Type::Object) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    std::string type;
    if (!readString(record, "type", &type, 32u, true, false)) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    if (type == "document") {
      if (documentSeen || !parseDocumentRecord(record, &parsed.workspace)) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      documentSeen = true;
    } else if (type == "presentation") {
      if (presentationSeen || !parsePresentationRecord(record, &parsed.presentation, true)) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      presentationSeen = true;
    } else if (type == "window") {
      PlotWindowDomainState window{};
      if (!parseWindow(record, &window, true)) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      if (parsed.workspace.windows.size() >= kViewerWorkspaceMaxWindows) {
        *status = ViewerWorkspacePersistenceStatus::CapacityExceeded;
        return false;
      }
      for (const auto& existing : parsed.workspace.windows) {
        if (existing.windowId == window.windowId) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
      }
      parsed.workspace.windows.push_back(std::move(window));
    } else {
      *status = ViewerWorkspacePersistenceStatus::UnknownRecord;
      return false;
    }
    lineStart = newline + 1u;
  }
  if (!documentSeen || !presentationSeen || parsed.workspace.windows.empty()) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  for (const auto& window : parsed.workspace.windows) {
    if (window.viewState.sourceSyncSelections !=
        parsed.workspace.sourceLassoSelectionsSynced) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
  }
  if (parsed.workspace.sourceLassoTargetWindowId == 0 ||
      parsed.workspace.sourceLassoTargetWindowId < -1 ||
      (parsed.workspace.sourceLassoSelectionsSynced &&
       parsed.workspace.sourceLassoTargetWindowId != -1)) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  if (!parsed.workspace.sourceLassoStrokes.empty() &&
      parsed.workspace.sourceLassoRevision == 0u) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  const bool globalLassoSelection = !parsed.workspace.sourceLassoStrokes.empty();
  bool activeLassoSelection = globalLassoSelection;
  if (!parsed.workspace.sourceLassoSelectionsSynced &&
      parsed.workspace.sourceLassoTargetWindowId > 0) {
    bool targetFound = false;
    for (const auto& window : parsed.workspace.windows) {
      if (window.windowId == parsed.workspace.sourceLassoTargetWindowId) {
        targetFound = true;
        activeLassoSelection = !window.viewerLassoStrokes.empty();
        break;
      }
    }
    if (!targetFound) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
  }
  if (parsed.workspace.sourceLassoGlobalHasSelection != globalLassoSelection ||
      parsed.workspace.sourceLassoHasSelection != activeLassoSelection) {
    *status = ViewerWorkspacePersistenceStatus::Malformed;
    return false;
  }
  for (const auto& window : parsed.workspace.windows) {
    const ViewerRuntimeState clamped = clampedViewerRuntimeState(window.viewState);
    if (!runtimeDurableEqual(window.viewState, clamped) ||
        window.viewState.sampleSettingsKey !=
            sampleSettingsKey(window.viewState, false)) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
  }
  const SanitiseResult sanitiseStatus = sanitiseDocumentInternal(parsed, output);
  if (sanitiseStatus != SanitiseResult::Accepted) {
    *status = sanitiseStatus == SanitiseResult::AllocationFailure
                  ? ViewerWorkspacePersistenceStatus::AllocationFailure
                  : ViewerWorkspacePersistenceStatus::ValidationFailed;
    return false;
  }
  *status = ViewerWorkspacePersistenceStatus::Accepted;
  return true;
}

bool legacyInt(const JsonValue& object, std::string_view name, int* output,
               int minimum, int maximum) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  if (value->type == JsonValue::Type::Bool) {
    if (!output) return false;
    *output = value->boolean ? 1 : 0;
    return *output >= minimum && *output <= maximum;
  }
  return readInt(object, name, output, minimum, maximum);
}

bool legacyUInt64(const JsonValue& object, std::string_view name,
                  uint64_t* output) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  return readUInt64(object, name, output, 0u,
                    std::numeric_limits<uint64_t>::max());
}

bool legacyBool(const JsonValue& object, std::string_view name, bool* output) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  if (value->type == JsonValue::Type::Bool) {
    if (!output) return false;
    *output = value->boolean;
    return true;
  }
  int parsed = 0;
  if (!legacyInt(object, name, &parsed, 0, 1) || !output) return false;
  *output = parsed != 0;
  return true;
}

bool legacyDouble(const JsonValue& object, std::string_view name, double* output,
                  double minimum = -std::numeric_limits<double>::max(),
                  double maximum = std::numeric_limits<double>::max()) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  return readDouble(object, name, output, minimum, maximum);
}

bool legacyFloat(const JsonValue& object, std::string_view name, float* output,
                 float minimum = -std::numeric_limits<float>::max(),
                 float maximum = std::numeric_limits<float>::max()) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  return readFloat(object, name, output, minimum, maximum);
}

bool legacyString(const JsonValue& object, std::string_view name,
                  std::string* output, std::size_t maximum) {
  const JsonValue* value = member(object, name);
  if (!value) return true;
  return readString(object, name, output, maximum);
}

bool parseLegacyRuntime(const JsonValue& object, ViewerRuntimeState* output) {
  if (!output || object.type != JsonValue::Type::Object) return false;
  ViewerRuntimeState parsed = *output;
    if (!legacyUInt64(object, "stateRevision", &parsed.stateRevision) ||
        !legacyString(object, "sampleSettingsKey", &parsed.sampleSettingsKey,
                      kViewerWorkspacePersistenceMaxRuntimeStringBytes) ||
        !legacyInt(object, "plotModel", &parsed.plotModel, 0, kPlotModelCount - 1) ||
        !legacyBool(object, "circularHsl", &parsed.circularHsl) ||
        !legacyBool(object, "circularHsv", &parsed.circularHsv) ||
        !legacyBool(object, "normConeNormalized", &parsed.normConeNormalized) ||
        !legacyBool(object, "plotDisplayLinear", &parsed.plotDisplayLinear) ||
        !legacyInt(object, "plotDisplayLinearTransfer", &parsed.plotDisplayLinearTransfer, 0, 4096) ||
        !legacyBool(object, "liveUpdate", &parsed.liveUpdate) ||
        !legacyInt(object, "updateMode", &parsed.updateMode, 0, 2) ||
        !legacyInt(object, "sourceDetailMode", &parsed.sourceDetailMode, 0, 4) ||
        !legacyInt(object, "sourceMaxProxyLongEdge", &parsed.sourceMaxProxyLongEdge, 0, 100000) ||
        !legacyBool(object, "sourceUseNativeWhenAvailable", &parsed.sourceUseNativeWhenAvailable) ||
        !legacyBool(object, "sourceSyncSelections", &parsed.sourceSyncSelections) ||
        !legacyBool(object, "sourceSyncCommonPlotSettings", &parsed.sourceSyncCommonPlotSettings) ||
        !legacyInt(object, "quality", &parsed.quality, 0, 100) ||
        !legacyInt(object, "scale", &parsed.scale, 0, 100) ||
        !legacyInt(object, "sampling", &parsed.sampling, 0, 100) ||
        !legacyBool(object, "occupancyFill", &parsed.occupancyGuidedFill) ||
        !legacyInt(object, "plotStyle", &parsed.plotStyle, 0, 100) ||
        !legacyInt(object, "pointShape", &parsed.pointShape, 0, 100) ||
        !legacyDouble(object, "pointSize", &parsed.pointSize) ||
        !legacyDouble(object, "colorSaturation", &parsed.colorSaturation) ||
        !legacyBool(object, "keepOnTop", &parsed.keepOnTop) ||
        !legacyBool(object, "resetViewOnPlotSwitch", &parsed.resetViewOnPlotSwitch) ||
        !legacyBool(object, "showOverflow", &parsed.showOverflow) ||
        !legacyBool(object, "highlightOverflow", &parsed.highlightOverflow) ||
        !legacyDouble(object, "overflowHighlightColorR", &parsed.overflowHighlightR) ||
        !legacyDouble(object, "overflowHighlightColorG", &parsed.overflowHighlightG) ||
        !legacyDouble(object, "overflowHighlightColorB", &parsed.overflowHighlightB) ||
        !legacyDouble(object, "viewerBackgroundColorR", &parsed.backgroundR) ||
        !legacyDouble(object, "viewerBackgroundColorG", &parsed.backgroundG) ||
        !legacyDouble(object, "viewerBackgroundColorB", &parsed.backgroundB) ||
        !legacyBool(object, "identityOverlayEnabled", &parsed.fillVolume) ||
        !legacyBool(object, "identityOverlayRamp", &parsed.fillGrayRamp) ||
        !legacyInt(object, "identityOverlayRequestedSize", &parsed.fillResolution, 0, 10000) ||
        !legacyBool(object, "readGrayRamp", &parsed.readGrayRamp) ||
        !legacyBool(object, "readIdentityPlot", &parsed.readIdentityPlot) ||
        !legacyBool(object, "isolateIdentityData", &parsed.isolateIdentityData) ||
        !legacyBool(object, "excludeIdentityData", &parsed.excludeIdentityData) ||
        !legacyInt(object, "identityReadResolution", &parsed.identityReadResolution, 0, 10000) ||
        !legacyBool(object, "volumeSliceLassoRegion", &parsed.volumeSliceLassoRegion) ||
        !legacyBool(object, "cubeSliceRed", &parsed.volumeSliceRed) ||
        !legacyBool(object, "cubeSliceYellow", &parsed.volumeSliceYellow) ||
        !legacyBool(object, "cubeSliceGreen", &parsed.volumeSliceGreen) ||
        !legacyBool(object, "cubeSliceCyan", &parsed.volumeSliceCyan) ||
        !legacyBool(object, "cubeSliceBlue", &parsed.volumeSliceBlue) ||
        !legacyBool(object, "cubeSliceMagenta", &parsed.volumeSliceMagenta) ||
        !legacyDouble(object, "neutralRadius", &parsed.neutralRadius) ||
        !legacyInt(object, "chromaticityInputPrimaries", &parsed.chromaticityInputPrimaries, 0, 100000) ||
        !legacyInt(object, "chromaticityInputTransfer", &parsed.chromaticityInputTransfer, 0, 100000) ||
        !legacyInt(object, "chromaticityReferenceBasis", &parsed.chromaticityReferenceBasis, 0, 100) ||
        !legacyInt(object, "chromaticityOverlayPrimaries", &parsed.chromaticityOverlayPrimaries, 0, 100000) ||
        !legacyBool(object, "chromaticityPlanckianLocus", &parsed.chromaticityPlanckianLocus) ||
        !legacyBool(object, "chromaticitySpectralLocus3D", &parsed.chromaticitySpectralLocus3D) ||
        !legacyInt(object, "glossNeighborhood", &parsed.glossNeighborhood, 0, 100) ||
        !legacyDouble(object, "glossLiftScale", &parsed.glossLiftScale) ||
        !legacyBool(object, "glossSpatialInset", &parsed.glossSpatialInset) ||
        !legacyDouble(object, "glossBodyOpacity", &parsed.glossBodyOpacity) ||
        !legacyDouble(object, "glossHighlightOpacity", &parsed.glossHighlightOpacity) ||
        !legacyDouble(object, "glossPointCrispness", &parsed.glossPointCrispness) ||
        !legacyBool(object, "glossHideText", &parsed.glossHideText) ||
        !legacyInt(object, "waveformMode", &parsed.waveformMode, 0, 100) ||
        !legacyBool(object, "waveformHighDetail", &parsed.waveformHighDetail) ||
        !legacyBool(object, "waveformContinuousHighDetail", &parsed.waveformContinuousHighDetail) ||
        !legacyBool(object, "waveformHighDetailRequested", &parsed.waveformHighDetailRequested) ||
        !legacyInt(object, "waveformSampleColumns", &parsed.waveformSampleColumns, 0, 100000) ||
        !legacyInt(object, "waveformSamplesPerColumn", &parsed.waveformSamplesPerColumn, 0, 100000) ||
        !legacyDouble(object, "waveformPointBrightness", &parsed.waveformPointBrightness) ||
        !legacyDouble(object, "waveformGridBrightness", &parsed.waveformGridBrightness) ||
        !legacyDouble(object, "waveformSaturation", &parsed.waveformSaturation) ||
        !legacyDouble(object, "waveformDotSize", &parsed.waveformDotSize) ||
        !legacyBool(object, "waveformChannelRed", &parsed.waveformChannelRed) ||
        !legacyBool(object, "waveformChannelGreen", &parsed.waveformChannelGreen) ||
        !legacyBool(object, "waveformChannelBlue", &parsed.waveformChannelBlue) ||
        !legacyBool(object, "waveformChannelLuma", &parsed.waveformChannelLuma) ||
        !legacyBool(object, "waveformShowOverflow", &parsed.waveformShowOverflow) ||
        !legacyBool(object, "waveformHighlightOverflow", &parsed.waveformHighlightOverflow) ||
        !legacyInt(object, "waveformLumaMethod", &parsed.waveformLumaMethod, 0, 100) ||
        !legacyInt(object, "histogramMode", &parsed.histogramMode, 0, 100) ||
        !legacyBool(object, "histogramShowOverflow", &parsed.histogramShowOverflow) ||
        !legacyBool(object, "histogramHighlightOverflow", &parsed.histogramHighlightOverflow) ||
        !legacyInt(object, "scopeRangeMode", &parsed.scopeRangeMode, 0, 100)) {
      return false;
    }
    // Old settings used aliases for these options; consume them only after
    // the common validation above so malformed values still reject.
    parsed = clampedViewerRuntimeState(parsed);
    parsed.refreshPolicy = "none";
    parsed.requiresHostSamples = false;
    parsed.sourceSessionId.clear();
    parsed.hostRefreshRequestedRevision = 0u;
    if (parsed.stateRevision == 0u) parsed.stateRevision = 1u;
    *output = parsed;
  return true;
}

bool parseLegacyLassoData(std::string_view data, PlotWindowDomainState* window) {
  if (!window || data.empty()) return true;
  if (data.size() > kViewerWorkspacePersistenceMaxLassoBytes ||
      data.substr(0u, 3u) != "v1|") {
    return false;
  }
  std::vector<LassoStroke> strokes;
    std::size_t cursor = 3u;
    const auto nextToken = [&](std::string_view* token) -> bool {
      if (!token || cursor > data.size()) return false;
      const std::size_t end = data.find('|', cursor);
      if (end == std::string_view::npos) {
        *token = data.substr(cursor);
        cursor = data.size();
      } else {
        *token = data.substr(cursor, end - cursor);
        cursor = end + 1u;
      }
      return true;
    };
    std::string_view token;
    if (!nextToken(&token)) return false;
    uint64_t revision = 0u;
    if (!parseUInt64(token, &revision)) return false;
    std::size_t totalPoints = 0u;
    while (cursor < data.size()) {
      if (!nextToken(&token) || token.size() < 3u || (token[0] != 'a' && token[0] != 's') ||
          token[1] != ',') return false;
      std::size_t start = 2u;
      const std::size_t comma = token.find(',', start);
      if (comma == std::string_view::npos) return false;
      uint64_t count = 0u;
      if (!parseUInt64(token.substr(start, comma - start), &count) ||
          count > kViewerWorkspaceMaxLassoPointsPerStroke ||
          count > kViewerWorkspaceMaxLassoPoints - totalPoints) {
        return false;
      }
      LassoStroke stroke{};
      stroke.subtract = token[0] == 's';
      start = comma + 1u;
      stroke.points.reserve(static_cast<std::size_t>(count));
      for (uint64_t i = 0u; i < count; ++i) {
        const std::size_t xComma = token.find(',', start);
        if (xComma == std::string_view::npos) return false;
        const std::size_t yEnd = token.find(',', xComma + 1u);
        const std::size_t end = yEnd == std::string_view::npos ? token.size() : yEnd;
        double x = 0.0;
        double y = 0.0;
        if (!parseDouble(token.substr(start, xComma - start), &x) ||
            !parseDouble(token.substr(xComma + 1u, end - xComma - 1u), &y)) {
          return false;
        }
        LassoPointNorm point{};
        point.xNorm = static_cast<float>(std::max(0.0, std::min(1.0, x)));
        point.yNorm = static_cast<float>(std::max(0.0, std::min(1.0, y)));
        stroke.points.push_back(point);
        start = yEnd == std::string_view::npos ? token.size() : yEnd + 1u;
      }
      if (start != token.size() || count < 3u) return false;
      totalPoints += static_cast<std::size_t>(count);
      recomputeBounds(&stroke);
      strokes.push_back(std::move(stroke));
    }
    window->viewerLassoRevision = revision;
    window->viewerLassoData.assign(data.data(), data.size());
    window->viewerLassoStrokes = std::move(strokes);
  return true;
}

void repairLegacyAnalyticalCamera(PlotWindowDomainState* window) noexcept {
  if (!window) return;
  const int model = window->viewState.plotModel;
  if (model != kPlotModelWaveform && model != kPlotModelHistogram) return;
  const bool generic = window->camera.orthographic &&
                       std::abs(window->camera.distance - 6.0f) < 1.0e-4f &&
                       std::abs(window->camera.panX) < 1.0e-4f &&
                       std::abs(window->camera.panY) < 1.0e-4f;
  if (!window->camera.orthographic || generic) {
    window->camera.orthographic = true;
    window->camera.orthographicView = -1;
    window->camera.distance = 6.35f;
    window->camera.panX = model == kPlotModelHistogram ? -0.07f : 0.035f;
    window->camera.panY = 0.0f;
    window->camera.qx = 0.0f;
    window->camera.qy = 0.0f;
    window->camera.qz = 0.0f;
    window->camera.qw = 1.0f;
  }
}

bool parseLegacy(std::string_view bytes, const ViewerWorkspaceDocument* defaults,
                 ViewerWorkspaceDocument* output,
                 ViewerWorkspacePersistenceStatus* status) {
  if (!output || !status) return false;
  try {
    ViewerWorkspaceDocument parsed = defaults ? *defaults : defaultViewerWorkspaceDocument();
    parsed.workspace.windows.clear();
    parsed.presentation = defaults ? defaults->presentation
                                   : defaultViewerWorkspaceDocument().presentation;
    parsed.presentation.windowWidth = 720;
    parsed.presentation.windowHeight = 600;
    parsed.presentation.workspaceTopNorm = 50.0f / 600.0f;
    int focused = 1;
    int nextId = 2;
    bool headerSeen = false;
    std::size_t lineStart = 0u;
    std::size_t lineCount = 0u;
    while (lineStart < bytes.size()) {
      const std::size_t newline = bytes.find('\n', lineStart);
      const std::size_t end = newline == std::string_view::npos ? bytes.size() : newline;
      if (++lineCount > kViewerWorkspacePersistenceMaxLines || end == lineStart) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      std::string_view line = bytes.substr(lineStart, end - lineStart);
      if (!line.empty() && line.back() == '\r') line.remove_suffix(1u);
      JsonValue record;
      if (!parseJson(line, &record) || record.type != JsonValue::Type::Object) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      std::string type;
      if (!readString(record, "type", &type, 64u, true, false)) {
        *status = ViewerWorkspacePersistenceStatus::Malformed;
        return false;
      }
      if (type == kViewerWorkspacePersistenceV1Type) {
        if (headerSeen) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
        headerSeen = true;
        if (!legacyInt(record, "focusedWindowId", &focused, 1,
                       std::numeric_limits<int>::max()) ||
            !legacyInt(record, "nextWindowId", &nextId, 1,
                       std::numeric_limits<int>::max()) ||
            !legacyBool(record, "showWorkspaceButtons",
                        &parsed.presentation.showWorkspaceButtons) ||
            !legacyBool(record, "showSliceButtonInPlotWindows",
                        &parsed.presentation.showSliceButtonInPlotWindows) ||
            !legacyInt(record, "viewerFontSize", &parsed.presentation.viewerFontSize, -100, 100) ||
            !legacyInt(record, "windowWidth", &parsed.presentation.windowWidth, -100000, 100000) ||
            !legacyInt(record, "windowHeight", &parsed.presentation.windowHeight, -100000, 100000) ||
            !legacyInt(record, "windowPosX", &parsed.presentation.windowPosX,
                       std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) ||
            !legacyInt(record, "windowPosY", &parsed.presentation.windowPosY,
                       std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) ||
            !legacyBool(record, "windowPositionValid", &parsed.presentation.windowPositionValid) ||
            !legacyInt(record, "activeStandardLayoutIndex",
                       &parsed.presentation.activeStandardLayoutIndex, -100, 100) ||
            !legacyFloat(record, "workspaceTopNorm", &parsed.presentation.workspaceTopNorm,
                         -100.0f, 100.0f) ||
            !legacyString(record, "layoutPresetSelection", &parsed.workspace.layoutPresetSelection,
                          kViewerWorkspaceMaxStringBytes)) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
        if (parsed.workspace.layoutPresetSelection.empty()) {
          parsed.workspace.layoutPresetSelection = "Single";
        }
      } else if (type == "plot_window") {
        if (!headerSeen || parsed.workspace.windows.size() >= kViewerWorkspaceMaxWindows) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
        PlotWindowDomainState window{};
        const char* requiredLegacyWindowFields[] = {
            "windowId", "x", "y", "w", "h", "camQx", "camQy", "camQz",
            "camQw", "camDistance", "camPanX", "camPanY", "camOrthographic",
            "camOrthographicView", "plotModel"};
        for (const char* field : requiredLegacyWindowFields) {
          if (!member(record, field)) {
            *status = ViewerWorkspacePersistenceStatus::Malformed;
            return false;
          }
        }
        if (!legacyInt(record, "windowId", &window.windowId, 1,
                       std::numeric_limits<int>::max()) ||
            !legacyFloat(record, "x", &window.rect.x) ||
            !legacyFloat(record, "y", &window.rect.y) ||
            !legacyFloat(record, "w", &window.rect.w) ||
            !legacyFloat(record, "h", &window.rect.h) ||
            !legacyFloat(record, "camQx", &window.camera.qx) ||
            !legacyFloat(record, "camQy", &window.camera.qy) ||
            !legacyFloat(record, "camQz", &window.camera.qz) ||
            !legacyFloat(record, "camQw", &window.camera.qw) ||
            !legacyFloat(record, "camDistance", &window.camera.distance) ||
            !legacyFloat(record, "camPanX", &window.camera.panX) ||
            !legacyFloat(record, "camPanY", &window.camera.panY) ||
            !legacyBool(record, "camOrthographic", &window.camera.orthographic) ||
            !legacyInt(record, "camOrthographicView", &window.camera.orthographicView,
                       -100, 100) ||
            !parseLegacyRuntime(record, &window.viewState)) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
        std::string lassoData;
        if (!legacyString(record, "viewerLassoData", &lassoData,
                          kViewerWorkspacePersistenceMaxLassoBytes) ||
            (!lassoData.empty() && !parseLegacyLassoData(lassoData, &window))) {
          *status = ViewerWorkspacePersistenceStatus::Malformed;
          return false;
        }
        window.rect.x = std::max(0.0f, std::min(1.0f, window.rect.x));
        window.rect.y = std::max(0.0f, std::min(1.0f, window.rect.y));
        window.rect.w = std::max(0.05f, std::min(1.0f, window.rect.w));
        window.rect.h = std::max(0.05f, std::min(1.0f, window.rect.h));
        if (window.rect.x + window.rect.w > 1.0f) window.rect.x = 1.0f - window.rect.w;
        if (window.rect.y + window.rect.h > 1.0f) window.rect.y = 1.0f - window.rect.h;
        repairLegacyAnalyticalCamera(&window);
        for (const auto& existing : parsed.workspace.windows) {
          if (existing.windowId == window.windowId) {
            *status = ViewerWorkspacePersistenceStatus::Malformed;
            return false;
          }
        }
        parsed.workspace.windows.push_back(std::move(window));
      } else {
        *status = ViewerWorkspacePersistenceStatus::UnknownRecord;
        return false;
      }
      if (newline == std::string_view::npos) break;
      lineStart = newline + 1u;
    }
    if (!headerSeen || parsed.workspace.windows.empty()) {
      *status = ViewerWorkspacePersistenceStatus::Malformed;
      return false;
    }
    int maxId = 0;
    for (auto& window : parsed.workspace.windows) maxId = std::max(maxId, window.windowId);
    if (focused <= 0) focused = parsed.workspace.windows.front().windowId;
    bool focusExists = false;
    for (const auto& window : parsed.workspace.windows) focusExists |= window.windowId == focused;
    if (!focusExists) focused = parsed.workspace.windows.front().windowId;
    if (nextId <= maxId) {
      if (maxId == std::numeric_limits<int>::max()) {
        *status = ViewerWorkspacePersistenceStatus::ValidationFailed;
        return false;
      }
      nextId = maxId + 1;
    }
    parsed.workspace.focusedWindowId = focused;
    parsed.workspace.nextWindowId = nextId;
    parsed.workspace.revision = 1u;
    parsed.workspace.layoutPresetBeforeSolo.clear();
    parsed.workspace.layoutPresetNameInput.clear();
    parsed.presentation.viewerFontSize = std::max(0, std::min(2, parsed.presentation.viewerFontSize));
    parsed.presentation.windowWidth = std::max(0, std::min(16384, parsed.presentation.windowWidth));
    parsed.presentation.windowHeight = std::max(0, std::min(16384, parsed.presentation.windowHeight));
    parsed.presentation.activeStandardLayoutIndex = std::max(
        0, std::min(kViewerLayoutChoiceCount - 1, parsed.presentation.activeStandardLayoutIndex));
    parsed.presentation.workspaceTopNorm = std::max(0.0f, std::min(1.0f, parsed.presentation.workspaceTopNorm));
    const SanitiseResult sanitiseStatus = sanitiseDocumentInternal(parsed, output);
    if (sanitiseStatus != SanitiseResult::Accepted) {
      *status = sanitiseStatus == SanitiseResult::AllocationFailure
                    ? ViewerWorkspacePersistenceStatus::AllocationFailure
                    : ViewerWorkspacePersistenceStatus::ValidationFailed;
      return false;
    }
    *status = ViewerWorkspacePersistenceStatus::Accepted;
    return true;
  } catch (...) {
    *status = ViewerWorkspacePersistenceStatus::AllocationFailure;
    return false;
  }
}

bool resetTransientFields(ViewerWorkspaceDocument* document) {
  if (!document) return false;
  ViewerWorkspaceState& state = document->workspace;
  if (!validPersistedLassoStrokes(state.sourceLassoStrokes)) return false;
  state.activeToolbarPanel = ViewerWorkspaceToolbarPanel::None;
  state.toolbarPanelAnchorX = 0.0f;
  state.toolbarPanelAnchorY = 0.0f;
  state.windowDragActive = false;
  state.windowDragWindowId = -1;
  state.windowDragMode = PlotWindowDragMode::None;
  state.windowDragStartX = 0.0f;
  state.windowDragStartY = 0.0f;
  state.windowDragStartRect = {};
  state.sourceLassoSubtractMode = false;
  state.sourceLassoSessionActive = false;
  for (auto& stroke : state.sourceLassoStrokes) recomputeBounds(&stroke);
  const bool hasSourceSelection = !state.sourceLassoStrokes.empty();
  state.sourceLassoGlobalHasSelection = hasSourceSelection;
  if (state.sourceLassoRevision == 0u && hasSourceSelection) state.sourceLassoRevision = 1u;
  bool targetValid = state.sourceLassoTargetWindowId > 0;
  if (targetValid) {
    targetValid = false;
    for (const auto& window : state.windows) {
      if (window.windowId == state.sourceLassoTargetWindowId) {
        targetValid = true;
        break;
      }
    }
  }
  if (state.sourceLassoSelectionsSynced || !targetValid) {
    state.sourceLassoTargetWindowId = -1;
  }
  if (state.sourceLassoSelectionsSynced || state.sourceLassoTargetWindowId <= 0) {
    state.sourceLassoHasSelection = hasSourceSelection;
  } else {
    state.sourceLassoHasSelection = false;
    for (const auto& window : state.windows) {
      if (window.windowId == state.sourceLassoTargetWindowId) {
        state.sourceLassoHasSelection = !window.viewerLassoStrokes.empty();
        break;
      }
    }
  }
  if (state.layoutPresetSelection.empty()) state.layoutPresetSelection = "Custom";
  for (auto& window : state.windows) {
    if (!validPersistedLassoStrokes(window.viewerLassoStrokes)) return false;
    window.selected = window.windowId == state.focusedWindowId;
    window.fitRequested = false;
    window.syncLabel = "Waiting for Resolve";
    window.stableSyncLabel = "Waiting for Resolve";
    window.lastHealthySyncLabelTime = -10.0;
    window.sourceSignalDocked = false;
    window.sourceSignalTemporaryLassoSurface = false;
    window.sourceSignalDockOwnerWindowId = -1;
    window.sourceSignalRestoreRect = {};
    window.sourceSignalDockAnimStart = -10.0;
    window.sourceSignalDockAnimatingToDock = false;
    window.slicingDrawerAnimStart = -10.0;
    window.viewState.refreshPolicy = "none";
    window.viewState.requiresHostSamples = false;
    window.viewState.sourceSessionId.clear();
    window.viewState.hostRefreshRequestedRevision = 0u;
    window.viewState.sourceSyncSelections = state.sourceLassoSelectionsSynced;
    window.viewState = clampedViewerRuntimeState(window.viewState);
    for (auto& stroke : window.viewerLassoStrokes) recomputeBounds(&stroke);
    if (!window.viewerLassoStrokes.empty() && window.viewerLassoRevision == 0u) {
      window.viewerLassoRevision = 1u;
    }
    window.viewerLassoData = canonicalLassoData(window.viewerLassoRevision,
                                                window.viewerLassoStrokes);
    window.viewState.sampleSettingsKey =
        sampleSettingsKey(window.viewState, false);
  }
  return true;
}

bool validPresentationInternal(
    const ViewerWorkspacePresentationPreferences& prefs) noexcept {
  return prefs.viewerFontSize >= 0 && prefs.viewerFontSize <= 2 &&
         prefs.windowWidth >= 0 && prefs.windowWidth <= 16384 &&
         prefs.windowHeight >= 0 && prefs.windowHeight <= 16384 &&
         prefs.activeStandardLayoutIndex >= 0 &&
         prefs.activeStandardLayoutIndex < kViewerLayoutChoiceCount &&
         finite(prefs.workspaceTopNorm) && prefs.workspaceTopNorm >= 0.0f &&
         prefs.workspaceTopNorm <= 1.0f;
}

bool validateDocumentThrowing(const ViewerWorkspaceDocument& document) {
  if (!validPresentationInternal(document.presentation) ||
      !validateViewerWorkspaceState(document.workspace)) {
    return false;
  }
  std::size_t serializedBytes = 0u;
  if (!validPersistedLassoStrokes(document.workspace.sourceLassoStrokes)) {
    return false;
  }
  const bool hasSourceSelection = !document.workspace.sourceLassoStrokes.empty();
  if (document.workspace.sourceLassoTargetWindowId < -1 ||
      document.workspace.sourceLassoGlobalHasSelection != hasSourceSelection ||
      (hasSourceSelection && document.workspace.sourceLassoRevision == 0u) ||
      (document.workspace.sourceLassoSelectionsSynced &&
       document.workspace.sourceLassoTargetWindowId != -1)) {
    return false;
  }
  const int sourceTargetId = document.workspace.sourceLassoTargetWindowId;
  bool targetHasSelection = hasSourceSelection;
  if (document.workspace.sourceLassoTargetWindowId > 0) {
    bool foundTarget = false;
    for (const auto& window : document.workspace.windows) {
      if (window.windowId == sourceTargetId) {
        foundTarget = true;
        targetHasSelection = !window.viewerLassoStrokes.empty();
        break;
      }
    }
    if (!foundTarget) return false;
  }
  const bool expectedActiveSelection =
      (document.workspace.sourceLassoSelectionsSynced || sourceTargetId <= 0)
          ? hasSourceSelection
          : targetHasSelection;
  if (document.workspace.sourceLassoHasSelection != expectedActiveSelection) {
    return false;
  }
  for (const auto& window : document.workspace.windows) {
    if (!validPersistedLassoStrokes(window.viewerLassoStrokes)) return false;
    const std::string canonicalData =
        canonicalLassoData(window.viewerLassoRevision, window.viewerLassoStrokes);
    if (!finiteCamera(window.camera) || !finiteRect(window.rect) ||
        window.viewState.stateRevision == 0u ||
        window.viewState.sampleSettingsKey.size() >
            kViewerWorkspacePersistenceMaxRuntimeStringBytes ||
        (!window.viewerLassoStrokes.empty() && window.viewerLassoRevision == 0u) ||
        window.viewerLassoData != canonicalData ||
        window.viewerLassoData.size() > kViewerWorkspacePersistenceMaxLassoBytes ||
        serializedBytes > kViewerWorkspacePersistenceMaxLassoBytes -
                              window.viewerLassoData.size()) {
      return false;
    }
    serializedBytes += window.viewerLassoData.size();
    const ViewerRuntimeState clamped = clampedViewerRuntimeState(window.viewState);
    if (!runtimeDurableEqual(window.viewState, clamped) ||
        window.viewState.sampleSettingsKey != sampleSettingsKey(window.viewState, false) ||
        window.viewState.sourceSyncSelections !=
            document.workspace.sourceLassoSelectionsSynced ||
        window.viewState.refreshPolicy != "none" ||
        window.viewState.requiresHostSamples || !window.viewState.sourceSessionId.empty() ||
        window.viewState.hostRefreshRequestedRevision != 0u) {
      return false;
    }
  }
  return true;
}

SanitiseResult sanitiseDocumentInternal(const ViewerWorkspaceDocument& input,
                                        ViewerWorkspaceDocument* output) {
  if (!output) return SanitiseResult::Invalid;
  try {
    ViewerWorkspaceDocument sanitized = input;
    if (!resetTransientFields(&sanitized) ||
        !validateDocumentThrowing(sanitized)) {
      return SanitiseResult::Invalid;
    }
    *output = std::move(sanitized);
    return SanitiseResult::Accepted;
  } catch (const std::bad_alloc&) {
    return SanitiseResult::AllocationFailure;
  } catch (...) {
    return SanitiseResult::Invalid;
  }
}

}  // namespace

ViewerWorkspaceDocument defaultViewerWorkspaceDocument() noexcept {
  try {
    ViewerWorkspaceDocument document{};
    document.workspace.windows.clear();
    PlotWindowDomainState window{};
    window.windowId = 1;
    window.rect = {};
    window.selected = true;
    window.syncLabel = "Waiting for Resolve";
    window.stableSyncLabel = "Waiting for Resolve";
    window.lastHealthySyncLabelTime = -10.0;
    document.workspace.windows.push_back(std::move(window));
    document.workspace.focusedWindowId = 1;
    document.workspace.nextWindowId = 2;
    document.workspace.layoutPresetSelection = "Single";
    document.workspace.layoutPresetBeforeSolo.clear();
    document.workspace.layoutPresetNameInput.clear();
    document.workspace.revision = 1u;
    document.presentation = ViewerWorkspacePresentationPreferences{};
    return document;
  } catch (...) {
    // The value-initialized object remains bounded and is used only as a
    // failure sentinel by callers; no exception crosses this API boundary.
    return ViewerWorkspaceDocument{};
  }
}

bool validateViewerWorkspaceDocument(const ViewerWorkspaceDocument& document) noexcept {
  try {
    return validateDocumentThrowing(document);
  } catch (...) {
    return false;
  }
}

bool sanitiseViewerWorkspaceDocument(const ViewerWorkspaceDocument& input,
                                     ViewerWorkspaceDocument* output) noexcept {
  try {
    return sanitiseDocumentInternal(input, output) == SanitiseResult::Accepted;
  } catch (...) {
    return false;
  }
}

ViewerWorkspacePersistenceEncodeResult encodeViewerWorkspaceV2(
    const ViewerWorkspaceDocument& document) noexcept {
  ViewerWorkspacePersistenceEncodeResult result{};
  try {
    ViewerWorkspaceDocument sanitized{};
    const SanitiseResult sanitiseStatus = sanitiseDocumentInternal(document, &sanitized);
    if (sanitiseStatus != SanitiseResult::Accepted) {
      result.status = sanitiseStatus == SanitiseResult::AllocationFailure
                          ? ViewerWorkspacePersistenceStatus::AllocationFailure
                          : ViewerWorkspacePersistenceStatus::ValidationFailed;
      return result;
    }
    if (sanitized.workspace.windows.size() + 2u > kViewerWorkspacePersistenceMaxLines) {
      result.status = ViewerWorkspacePersistenceStatus::CapacityExceeded;
      return result;
    }
    std::string payload;
    payload.reserve(4096u);
    std::string line;
    appendDocumentRecord(&line, sanitized.workspace);
    line.push_back('\n');
    payload.append(line);
    line.clear();
    appendPresentationRecord(&line, sanitized.presentation);
    line.push_back('\n');
    payload.append(line);
    for (const auto& window : sanitized.workspace.windows) {
      line.clear();
      appendWindow(&line, window);
      line.push_back('\n');
      payload.append(line);
      if (payload.size() > kViewerWorkspacePersistenceMaxDocumentBytes) {
        result.status = ViewerWorkspacePersistenceStatus::CapacityExceeded;
        return result;
      }
    }
    std::string header;
    beginObject(&header);
    appendString(&header, "schema", kViewerWorkspacePersistenceSchema);
    appendUInt64(&header, "version", kViewerWorkspacePersistenceVersion);
    appendUInt64(&header, "payloadBytes", payload.size());
    appendString(&header, "checksum", checksumString(fnv1a(payload)));
    endObject(&header);
    result.bytes.reserve(header.size() + 1u + payload.size());
    result.bytes = std::move(header);
    result.bytes.push_back('\n');
    result.bytes += payload;
    if (result.bytes.size() > kViewerWorkspacePersistenceMaxDocumentBytes) {
      result.bytes.clear();
      result.status = ViewerWorkspacePersistenceStatus::CapacityExceeded;
      return result;
    }
    result.status = ViewerWorkspacePersistenceStatus::Accepted;
    return result;
  } catch (...) {
    result.bytes.clear();
    result.status = ViewerWorkspacePersistenceStatus::AllocationFailure;
    return result;
  }
}

ViewerWorkspacePersistenceDecodeResult decodeViewerWorkspaceDocument(
    std::string_view bytes, const ViewerWorkspaceDocument* defaults) noexcept {
  ViewerWorkspacePersistenceDecodeResult result{};
  if (bytes.empty()) {
    result.status = ViewerWorkspacePersistenceStatus::EmptyInput;
    return result;
  }
  if (bytes.size() > kViewerWorkspacePersistenceMaxDocumentBytes) {
    result.status = ViewerWorkspacePersistenceStatus::CapacityExceeded;
    return result;
  }
  try {
    const std::size_t newline = bytes.find('\n');
    if (newline == std::string_view::npos) {
      result.status = ViewerWorkspacePersistenceStatus::Malformed;
      return result;
    }
    JsonValue first;
    if (!parseJson(bytes.substr(0u, newline), &first) || first.type != JsonValue::Type::Object) {
      result.status = ViewerWorkspacePersistenceStatus::Malformed;
      return result;
    }
    std::string type;
    if (readString(first, "type", &type, 64u, true, false) &&
        type == kViewerWorkspacePersistenceV1Type) {
      parseLegacy(bytes, defaults, &result.document, &result.status);
      return result;
    }
    parseV2(bytes, &result.document, &result.status);
    return result;
  } catch (...) {
    result.status = ViewerWorkspacePersistenceStatus::AllocationFailure;
    return result;
  }
}

}  // namespace ChromaspaceViewer
