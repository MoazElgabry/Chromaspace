#include "ChromaspaceViewerLiveCommand.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <new>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ChromaspaceViewer {
namespace {

enum class JsonKind : uint8_t { String, Number, Boolean, Null, Object, Array };

struct JsonField {
  std::string name;
  JsonKind kind = JsonKind::Null;
  std::string text;
  bool boolean = false;
};

struct ParsedObject {
  std::vector<JsonField> fields;
  ViewerLiveCommandStatus status = ViewerLiveCommandStatus::Accepted;
  std::string reason;
};

bool appendUtf8(uint32_t codepoint, std::string* output,
                std::size_t maximumBytes) {
  if (!output || codepoint > 0x10ffffu ||
      (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
    return false;
  }
  const std::size_t bytes = codepoint <= 0x7fu     ? 1u
                            : codepoint <= 0x7ffu  ? 2u
                            : codepoint <= 0xffffu ? 3u
                                                   : 4u;
  if (output->size() > maximumBytes ||
      bytes > maximumBytes - output->size()) {
    return false;
  }
  if (codepoint <= 0x7fu) {
    output->push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ffu) {
    output->push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
    output->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
  } else if (codepoint <= 0xffffu) {
    output->push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
    output->push_back(
        static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
    output->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
  } else {
    output->push_back(static_cast<char>(0xf0u | (codepoint >> 18u)));
    output->push_back(
        static_cast<char>(0x80u | ((codepoint >> 12u) & 0x3fu)));
    output->push_back(
        static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
    output->push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
  }
  return true;
}

class JsonCursor final {
 public:
  explicit JsonCursor(std::string_view input) : input_(input) {}

  bool parseTopObject(ParsedObject* output) {
    if (!output) return fail(ViewerLiveCommandStatus::Invalid,
                             "json-output-missing");
    skipWhitespace();
    if (!consume('{')) return fail(ViewerLiveCommandStatus::Malformed,
                                   "json-root-not-object");
    skipWhitespace();
    if (consume('}')) {
      skipWhitespace();
      if (position_ != input_.size()) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-trailing-bytes");
      }
      return publish(output);
    }

    std::unordered_set<std::string> names;
    for (;;) {
      if (++memberCount_ > kViewerLiveCommandMaxJsonMembers) {
        return fail(ViewerLiveCommandStatus::Oversized,
                    "json-member-limit");
      }
      JsonField field{};
      if (!parseString(&field.name, kViewerLiveCommandMaxStringBytes)) {
        return false;
      }
      if (!names.insert(field.name).second) {
        return fail(ViewerLiveCommandStatus::DuplicateField,
                    "json-duplicate-field");
      }
      skipWhitespace();
      if (!consume(':')) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-missing-colon");
      }
      skipWhitespace();
      const std::size_t stringLimit =
          field.name == "lassoData" ? kViewerLiveCommandMaxLassoBytes
                                    : kViewerLiveCommandMaxStringBytes;
      if (!parseValue(&field, 1u, stringLimit)) return false;
      parsed_.push_back(std::move(field));
      skipWhitespace();
      if (consume('}')) break;
      if (!consume(',')) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-missing-comma");
      }
      skipWhitespace();
    }
    skipWhitespace();
    if (position_ != input_.size()) {
      return fail(ViewerLiveCommandStatus::Malformed,
                  "json-trailing-bytes");
    }
    return publish(output);
  }

 private:
  bool publish(ParsedObject* output) {
    output->fields = std::move(parsed_);
    output->status = ViewerLiveCommandStatus::Accepted;
    output->reason.clear();
    return true;
  }

  bool fail(ViewerLiveCommandStatus status, const char* reason) {
    status_ = status;
    reason_ = reason ? reason : "json-invalid";
    return false;
  }

  void skipWhitespace() {
    while (position_ < input_.size()) {
      const char c = input_[position_];
      if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
      ++position_;
    }
  }

  bool consume(char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  static int hexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
  }

  bool parseHex4(uint32_t* value) {
    if (!value || input_.size() - position_ < 4u) {
      return fail(ViewerLiveCommandStatus::Malformed,
                  "json-short-unicode-escape");
    }
    uint32_t parsed = 0u;
    for (int index = 0; index < 4; ++index) {
      const int digit = hexDigit(input_[position_++]);
      if (digit < 0) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-invalid-unicode-escape");
      }
      parsed = (parsed << 4u) | static_cast<uint32_t>(digit);
    }
    *value = parsed;
    return true;
  }

  bool parseString(std::string* output, std::size_t maximumBytes) {
    if (!output || !consume('"')) {
      return fail(ViewerLiveCommandStatus::Malformed,
                  "json-string-expected");
    }
    output->clear();
    while (position_ < input_.size()) {
      const unsigned char c =
          static_cast<unsigned char>(input_[position_++]);
      if (c == '"') return true;
      if (c < 0x20u) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-control-in-string");
      }
      if (c != '\\') {
        if (output->size() >= maximumBytes) {
          return fail(ViewerLiveCommandStatus::Oversized,
                      "json-string-limit");
        }
        output->push_back(static_cast<char>(c));
        continue;
      }
      if (position_ >= input_.size()) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-short-escape");
      }
      const char escaped = input_[position_++];
      char decoded = 0;
      switch (escaped) {
        case '"': decoded = '"'; break;
        case '\\': decoded = '\\'; break;
        case '/': decoded = '/'; break;
        case 'b': decoded = '\b'; break;
        case 'f': decoded = '\f'; break;
        case 'n': decoded = '\n'; break;
        case 'r': decoded = '\r'; break;
        case 't': decoded = '\t'; break;
        case 'u': {
          uint32_t first = 0u;
          if (!parseHex4(&first)) return false;
          uint32_t codepoint = first;
          if (first >= 0xd800u && first <= 0xdbffu) {
            if (input_.size() - position_ < 6u ||
                input_[position_] != '\\' ||
                input_[position_ + 1u] != 'u') {
              return fail(ViewerLiveCommandStatus::Malformed,
                          "json-unpaired-high-surrogate");
            }
            position_ += 2u;
            uint32_t second = 0u;
            if (!parseHex4(&second)) return false;
            if (second < 0xdc00u || second > 0xdfffu) {
              return fail(ViewerLiveCommandStatus::Malformed,
                          "json-invalid-surrogate-pair");
            }
            codepoint = 0x10000u + ((first - 0xd800u) << 10u) +
                        (second - 0xdc00u);
          } else if (first >= 0xdc00u && first <= 0xdfffu) {
            return fail(ViewerLiveCommandStatus::Malformed,
                        "json-unpaired-low-surrogate");
          }
          if (!appendUtf8(codepoint, output, maximumBytes)) {
            return fail(ViewerLiveCommandStatus::Oversized,
                        "json-string-limit");
          }
          continue;
        }
        default:
          return fail(ViewerLiveCommandStatus::Malformed,
                      "json-invalid-escape");
      }
      if (output->size() >= maximumBytes) {
        return fail(ViewerLiveCommandStatus::Oversized,
                    "json-string-limit");
      }
      output->push_back(decoded);
    }
    return fail(ViewerLiveCommandStatus::Malformed,
                "json-unterminated-string");
  }

  bool parseNumber(std::string* output) {
    if (!output) return false;
    const std::size_t begin = position_;
    if (position_ < input_.size() && input_[position_] == '-') ++position_;
    if (position_ >= input_.size()) {
      return fail(ViewerLiveCommandStatus::Malformed, "json-short-number");
    }
    if (input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-leading-zero");
      }
    } else if (input_[position_] >= '1' && input_[position_] <= '9') {
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    } else {
      return fail(ViewerLiveCommandStatus::Malformed,
                  "json-invalid-number");
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (fraction == position_) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-empty-fraction");
      }
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
      if (exponent == position_) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-empty-exponent");
      }
    }
    if (position_ - begin > 128u) {
      return fail(ViewerLiveCommandStatus::Oversized,
                  "json-number-limit");
    }
    output->assign(input_.substr(begin, position_ - begin));
    return true;
  }

  bool consumeLiteral(std::string_view literal) {
    if (input_.size() - position_ < literal.size() ||
        input_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  bool skipValue(std::size_t depth) {
    JsonField ignored{};
    return parseValue(&ignored, depth, kViewerLiveCommandMaxStringBytes);
  }

  bool skipObject(std::size_t depth) {
    if (depth > kViewerLiveCommandMaxJsonDepth) {
      return fail(ViewerLiveCommandStatus::Oversized, "json-depth-limit");
    }
    if (!consume('{')) return false;
    skipWhitespace();
    if (consume('}')) return true;
    std::unordered_set<std::string> names;
    for (;;) {
      if (++memberCount_ > kViewerLiveCommandMaxJsonMembers) {
        return fail(ViewerLiveCommandStatus::Oversized,
                    "json-member-limit");
      }
      std::string name;
      if (!parseString(&name, kViewerLiveCommandMaxStringBytes)) return false;
      if (!names.insert(name).second) {
        return fail(ViewerLiveCommandStatus::DuplicateField,
                    "json-duplicate-field");
      }
      skipWhitespace();
      if (!consume(':')) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-missing-colon");
      }
      skipWhitespace();
      if (!skipValue(depth + 1u)) return false;
      skipWhitespace();
      if (consume('}')) return true;
      if (!consume(',')) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-missing-comma");
      }
      skipWhitespace();
    }
  }

  bool skipArray(std::size_t depth) {
    if (depth > kViewerLiveCommandMaxJsonDepth) {
      return fail(ViewerLiveCommandStatus::Oversized, "json-depth-limit");
    }
    if (!consume('[')) return false;
    skipWhitespace();
    if (consume(']')) return true;
    std::size_t items = 0u;
    for (;;) {
      if (++items > kViewerLiveCommandMaxJsonArrayItems) {
        return fail(ViewerLiveCommandStatus::Oversized,
                    "json-array-limit");
      }
      if (!skipValue(depth + 1u)) return false;
      skipWhitespace();
      if (consume(']')) return true;
      if (!consume(',')) {
        return fail(ViewerLiveCommandStatus::Malformed,
                    "json-missing-comma");
      }
      skipWhitespace();
    }
  }

  bool parseValue(JsonField* field, std::size_t depth,
                  std::size_t stringLimit) {
    if (!field || position_ >= input_.size()) {
      return fail(ViewerLiveCommandStatus::Malformed,
                  "json-value-missing");
    }
    const char c = input_[position_];
    if (c == '"') {
      field->kind = JsonKind::String;
      return parseString(&field->text, stringLimit);
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      field->kind = JsonKind::Number;
      return parseNumber(&field->text);
    }
    if (consumeLiteral("true")) {
      field->kind = JsonKind::Boolean;
      field->boolean = true;
      return true;
    }
    if (consumeLiteral("false")) {
      field->kind = JsonKind::Boolean;
      field->boolean = false;
      return true;
    }
    if (consumeLiteral("null")) {
      field->kind = JsonKind::Null;
      return true;
    }
    if (c == '{') {
      field->kind = JsonKind::Object;
      return skipObject(depth + 1u);
    }
    if (c == '[') {
      field->kind = JsonKind::Array;
      return skipArray(depth + 1u);
    }
    return fail(ViewerLiveCommandStatus::Malformed,
                "json-invalid-value");
  }

 public:
  ViewerLiveCommandStatus status() const noexcept { return status_; }
  const std::string& reason() const noexcept { return reason_; }

 private:
  std::string_view input_;
  std::size_t position_ = 0u;
  std::size_t memberCount_ = 0u;
  std::vector<JsonField> parsed_;
  ViewerLiveCommandStatus status_ = ViewerLiveCommandStatus::Accepted;
  std::string reason_;
};

const JsonField* findField(const ParsedObject& object, std::string_view name) {
  const auto found = std::find_if(
      object.fields.begin(), object.fields.end(),
      [&](const JsonField& field) { return field.name == name; });
  return found == object.fields.end() ? nullptr : &*found;
}

bool parseUnsigned(const JsonField& field, uint64_t* output) {
  if (!output || field.kind != JsonKind::Number || field.text.empty() ||
      field.text.front() == '-' || field.text.find_first_of(".eE") !=
                                     std::string::npos) {
    return false;
  }
  uint64_t value = 0u;
  const auto result = std::from_chars(field.text.data(),
                                      field.text.data() + field.text.size(),
                                      value);
  if (result.ec != std::errc{} ||
      result.ptr != field.text.data() + field.text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool parseInteger(const JsonField& field, int* output) {
  if (!output || field.kind != JsonKind::Number || field.text.empty() ||
      field.text.find_first_of(".eE") != std::string::npos) {
    return false;
  }
  int value = 0;
  const auto result = std::from_chars(field.text.data(),
                                      field.text.data() + field.text.size(),
                                      value);
  if (result.ec != std::errc{} ||
      result.ptr != field.text.data() + field.text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool parseFiniteFloat(const JsonField& field, float* output) {
  if (!output || field.kind != JsonKind::Number || field.text.empty()) {
    return false;
  }
  std::size_t position = 0u;
  bool negative = false;
  if (field.text[position] == '-') {
    negative = true;
    if (++position == field.text.size()) return false;
  }
  long double mantissa = 0.0L;
  int keptDigits = 0;
  int decimalExponent = 0;
  bool afterDecimal = false;
  bool anyNonZero = false;
  bool significantStarted = false;
  while (position < field.text.size()) {
    const char c = field.text[position];
    if (c == '.') {
      afterDecimal = true;
      ++position;
      continue;
    }
    if (c == 'e' || c == 'E') break;
    const int digit = c - '0';
    if (digit < 0 || digit > 9) return false;
    anyNonZero = anyNonZero || digit != 0;
    if (!significantStarted && digit == 0) {
      if (afterDecimal) --decimalExponent;
      ++position;
      continue;
    }
    significantStarted = true;
    if (keptDigits < 19) {
      mantissa = mantissa * 10.0L + static_cast<long double>(digit);
      ++keptDigits;
      if (afterDecimal) --decimalExponent;
    } else if (!afterDecimal) {
      ++decimalExponent;
    }
    ++position;
  }
  int explicitExponent = 0;
  if (position < field.text.size()) {
    ++position;
    bool exponentNegative = false;
    if (position < field.text.size() &&
        (field.text[position] == '+' || field.text[position] == '-')) {
      exponentNegative = field.text[position] == '-';
      ++position;
    }
    if (position == field.text.size()) return false;
    while (position < field.text.size()) {
      const int digit = field.text[position++] - '0';
      if (digit < 0 || digit > 9 || explicitExponent > 100000) return false;
      explicitExponent = explicitExponent * 10 + digit;
    }
    if (exponentNegative) explicitExponent = -explicitExponent;
  }
  const int totalExponent = decimalExponent + explicitExponent;
  if (totalExponent < -100000 || totalExponent > 100000) return false;
  long double value =
      mantissa * std::pow(10.0L, static_cast<long double>(totalExponent));
  if (negative) value = -value;
  if (!std::isfinite(value) ||
      value < -static_cast<long double>(std::numeric_limits<float>::max()) ||
      value > static_cast<long double>(std::numeric_limits<float>::max())) {
    return false;
  }
  *output = static_cast<float>(value);
  return std::isfinite(*output) && (!anyNonZero || *output != 0.0f);
}

bool parseWireBool(const JsonField& field, bool* output) {
  if (!output) return false;
  if (field.kind == JsonKind::Boolean) {
    *output = field.boolean;
    return true;
  }
  int value = 0;
  if (!parseInteger(field, &value) || (value != 0 && value != 1)) return false;
  *output = value != 0;
  return true;
}

bool requiredString(const ParsedObject& object, std::string_view name,
                    std::string* output, std::string* error) {
  const JsonField* field = findField(object, name);
  if (!field || field->kind != JsonKind::String || field->text.empty()) {
    if (error) *error = std::string("invalid-") + std::string(name);
    return false;
  }
  *output = field->text;
  return true;
}

bool requiredSequence(const ParsedObject& object, uint64_t* output,
                      std::string* error) {
  const JsonField* field = findField(object, "seq");
  if (!field || !parseUnsigned(*field, output) || *output == 0u) {
    if (error) *error = "invalid-seq";
    return false;
  }
  return true;
}

template <typename Value, typename Parse>
bool optionalField(const ParsedObject& object, std::string_view name,
                   Value* output, Parse parse, std::string* error) {
  const JsonField* field = findField(object, name);
  if (!field) return true;
  if (!parse(*field, output)) {
    if (error) *error = std::string("invalid-") + std::string(name);
    return false;
  }
  return true;
}

bool optionalString(const ParsedObject& object, std::string_view name,
                    std::string* output, std::string* error) {
  const JsonField* field = findField(object, name);
  if (!field) return true;
  if (field->kind != JsonKind::String) {
    if (error) *error = std::string("invalid-") + std::string(name);
    return false;
  }
  *output = field->text;
  return true;
}

int qualityIndex(std::string_view value) {
  if (value == "Medium") return 1;
  if (value == "High") return 2;
  return 0;
}

int scaleIndex(std::string_view value) {
  if (value == "25%") return 0;
  if (value == "50%") return 1;
  if (value == "75%") return 2;
  return 3;
}

int samplingIndex(std::string_view value) {
  if (value == "Stratified") return 1;
  if (value == "Random") return 2;
  return 0;
}

bool decodeParams(const ParsedObject& object, ViewerLiveCommandParams* params,
                  std::string* error) {
  if (!params || !requiredSequence(object, &params->seq, error) ||
      !requiredString(object, "senderId", &params->senderId, error)) {
    return false;
  }

#define READ_STRING(name, member) \
  if (!optionalString(object, name, &params->member, error)) return false
#define READ_INT(name, member)                                                \
  if (!optionalField(object, name, &params->member, parseInteger, error))     \
  return false
#define READ_U64(name, member)                                                \
  if (!optionalField(object, name, &params->member, parseUnsigned, error))    \
  return false
#define READ_FLOAT(name, member)                                              \
  if (!optionalField(object, name, &params->member, parseFiniteFloat, error)) \
  return false
#define READ_BOOL(name, member)                                               \
  if (!optionalField(object, name, &params->member, parseWireBool, error))    \
  return false

  READ_U64("stateRevision", stateRevision);
  READ_STRING("sourceMode", sourceMode);
  READ_BOOL("drawOnImageMode", drawOnImageMode);
  READ_STRING("plotMode", plotMode);
  READ_STRING("cloudSettingsKey", cloudSettingsKey);
  READ_BOOL("volumeSlicingEnabled", volumeSlicingEnabled);
  READ_STRING("volumeSlicingMode", volumeSlicingMode);
  READ_BOOL("lassoRegionEmpty", lassoRegionEmpty);
  READ_STRING("lassoData", lassoData);
  READ_BOOL("circularHsl", circularHsl);
  READ_BOOL("circularHsv", circularHsv);
  READ_BOOL("normConeNormalized", normConeNormalized);
  READ_BOOL("plotDisplayLinear", plotDisplayLinear);
  READ_INT("plotDisplayLinearTransfer", plotDisplayLinearTransfer);
  READ_FLOAT("sourceAspect", sourceAspect);
  READ_BOOL("alwaysOnTop", alwaysOnTop);
  READ_BOOL("resetViewOnPlotSwitch", resetViewOnPlotSwitch);
  READ_STRING("quality", quality);
  READ_STRING("sampling", sampling);
  READ_BOOL("occupancyFill", occupancyFill);
  READ_STRING("scale", scale);
  READ_INT("resolution", resolution);
  READ_FLOAT("pointSize", pointSize);
  READ_FLOAT("pointDensity", pointDensity);
  READ_FLOAT("colorSaturation", colorSaturation);
  READ_STRING("plotStyle", plotStyle);
  READ_STRING("pointShape", pointShape);
  READ_INT("glossNeighborhood", glossNeighborhood);
  READ_FLOAT("glossLiftScale", glossLiftScale);
  READ_BOOL("glossSpatialInset", glossSpatialInset);
  READ_FLOAT("glossBodyOpacity", glossBodyOpacity);
  READ_FLOAT("glossHighlightOpacity", glossHighlightOpacity);
  READ_FLOAT("glossPointCrispness", glossPointCrispness);
  READ_BOOL("glossHideText", glossHideText);
  READ_BOOL("showOverflow", showOverflow);
  READ_BOOL("highlightOverflow", highlightOverflow);
  READ_BOOL("cubeSlicingEnabled", cubeSlicingEnabled);
  READ_BOOL("neutralRadiusEnabled", neutralRadiusEnabled);
  READ_FLOAT("neutralRadius", neutralRadius);
  READ_BOOL("cubeSliceRed", cubeSliceRed);
  READ_BOOL("cubeSliceGreen", cubeSliceGreen);
  READ_BOOL("cubeSliceBlue", cubeSliceBlue);
  READ_BOOL("cubeSliceCyan", cubeSliceCyan);
  READ_BOOL("cubeSliceYellow", cubeSliceYellow);
  READ_BOOL("cubeSliceMagenta", cubeSliceMagenta);
  READ_FLOAT("overflowHighlightColorR", overflowHighlightR);
  READ_FLOAT("overflowHighlightColorG", overflowHighlightG);
  READ_FLOAT("overflowHighlightColorB", overflowHighlightB);
  READ_FLOAT("viewerBackgroundColorR", backgroundColorR);
  READ_FLOAT("viewerBackgroundColorG", backgroundColorG);
  READ_FLOAT("viewerBackgroundColorB", backgroundColorB);
  READ_BOOL("identityOverlayEnabled", identityOverlayEnabled);
  READ_BOOL("identityOverlayRamp", identityOverlayRamp);
  READ_BOOL("identityOverlayAuto", identityOverlayAuto);
  READ_INT("identityOverlayRequestedSize", identityOverlayRequestedSize);
  READ_INT("identityOverlaySize", identityOverlaySize);
  READ_BOOL("readGrayRamp", readGrayRamp);
  READ_BOOL("readIdentityPlot", readIdentityPlot);
  READ_BOOL("isolateIdentityData", isolateIdentityData);
  {
    const JsonField* exclude = findField(object, "excludeIdentityData");
    if (exclude) {
      params->hasExcludeIdentityData = true;
      if (!parseWireBool(*exclude, &params->excludeIdentityData)) {
        if (error) *error = "invalid-excludeIdentityData";
        return false;
      }
    }
  }
  READ_INT("identityReadResolution", identityReadResolution);
  READ_INT("generatedIdentityResolution", generatedIdentityResolution);
  READ_BOOL("generatedIdentityDrawCube", generatedIdentityDrawCube);
  READ_BOOL("generatedIdentityDrawRamp", generatedIdentityDrawRamp);
  READ_INT("generatedIdentityStripBandCount", generatedIdentityStripBandCount);
  READ_U64("generatedIdentityStripRevision", generatedIdentityStripRevision);
  READ_INT("chromaticityInputPrimaries", chromaticityInputPrimaries);
  READ_INT("chromaticityInputTransfer", chromaticityInputTransfer);
  READ_INT("chromaticityReferenceBasis", chromaticityReferenceBasis);
  READ_INT("chromaticityOverlayPrimaries", chromaticityOverlayPrimaries);
  READ_BOOL("chromaticityPlanckianLocus", chromaticityPlanckianLocus);
  READ_BOOL("chromaticitySpectralLocus3D", chromaticitySpectralLocus3D);
  READ_STRING("version", version);

  ViewerRuntimeState state{};
  state.stateRevision = params->stateRevision;
  state.plotModel = plotModelForMode(params->plotMode);
  state.circularHsl = params->circularHsl;
  state.circularHsv = params->circularHsv;
  state.normConeNormalized = params->normConeNormalized;
  state.plotDisplayLinear = params->plotDisplayLinear;
  state.plotDisplayLinearTransfer = params->plotDisplayLinearTransfer;

  READ_BOOL("liveUpdate", viewerState.liveUpdate);
  READ_INT("updateMode", viewerState.updateMode);
  READ_BOOL("occupancyGuidedFill", viewerState.occupancyGuidedFill);
  READ_INT("waveformMode", viewerState.waveformMode);
  READ_BOOL("waveformHighDetail", viewerState.waveformHighDetail);
  READ_BOOL("waveformContinuousHighDetail",
            viewerState.waveformContinuousHighDetail);
  READ_BOOL("waveformHighDetailRequested",
            viewerState.waveformHighDetailRequested);
  READ_INT("waveformSampleColumns", viewerState.waveformSampleColumns);
  READ_INT("waveformSamplesPerColumn", viewerState.waveformSamplesPerColumn);
  {
    float value = static_cast<float>(params->viewerState.waveformPointBrightness);
    if (!optionalField(object, "waveformPointBrightness", &value,
                       parseFiniteFloat, error)) return false;
    params->viewerState.waveformPointBrightness = value;
    value = static_cast<float>(params->viewerState.waveformGridBrightness);
    if (!optionalField(object, "waveformGridBrightness", &value,
                       parseFiniteFloat, error)) return false;
    params->viewerState.waveformGridBrightness = value;
    value = static_cast<float>(params->viewerState.waveformSaturation);
    if (!optionalField(object, "waveformSaturation", &value,
                       parseFiniteFloat, error)) return false;
    params->viewerState.waveformSaturation = value;
    value = static_cast<float>(params->viewerState.waveformDotSize);
    if (!optionalField(object, "waveformDotSize", &value,
                       parseFiniteFloat, error)) return false;
    params->viewerState.waveformDotSize = value;
  }
  READ_BOOL("waveformChannelRed", viewerState.waveformChannelRed);
  READ_BOOL("waveformChannelGreen", viewerState.waveformChannelGreen);
  READ_BOOL("waveformChannelBlue", viewerState.waveformChannelBlue);
  READ_BOOL("waveformChannelLuma", viewerState.waveformChannelLuma);
  READ_BOOL("waveformShowOverflow", viewerState.waveformShowOverflow);
  READ_BOOL("waveformHighlightOverflow",
            viewerState.waveformHighlightOverflow);
  READ_INT("waveformLumaMethod", viewerState.waveformLumaMethod);
  READ_INT("histogramMode", viewerState.histogramMode);
  READ_BOOL("histogramShowOverflow", viewerState.histogramShowOverflow);
  READ_BOOL("histogramHighlightOverflow",
            viewerState.histogramHighlightOverflow);
  READ_INT("scopeRangeMode", viewerState.scopeRangeMode);
  READ_INT("sourceDetailMode", viewerState.sourceDetailMode);
  READ_INT("sourceMaxProxyLongEdge", viewerState.sourceMaxProxyLongEdge);
  READ_BOOL("sourceUseNativeWhenAvailable",
            viewerState.sourceUseNativeWhenAvailable);
  READ_BOOL("sourceSyncSelections", viewerState.sourceSyncSelections);
  READ_BOOL("sourceSyncCommonPlotSettings",
            viewerState.sourceSyncCommonPlotSettings);

  state.liveUpdate = params->viewerState.liveUpdate;
  state.updateMode = params->viewerState.updateMode;
  state.quality = qualityIndex(params->quality);
  state.scale = scaleIndex(params->scale);
  state.sampling = samplingIndex(params->sampling);
  state.occupancyGuidedFill = params->viewerState.occupancyGuidedFill;
  state.plotStyle = params->plotStyle == "Space" ? 1 : 0;
  state.pointShape = params->pointShape == "Square" ? 1 : 0;
  state.pointSize = params->pointSize;
  state.colorSaturation = params->colorSaturation;
  state.keepOnTop = params->alwaysOnTop;
  state.resetViewOnPlotSwitch = params->resetViewOnPlotSwitch;
  state.showOverflow = params->showOverflow;
  state.highlightOverflow = params->highlightOverflow;
  state.overflowHighlightR = params->overflowHighlightR;
  state.overflowHighlightG = params->overflowHighlightG;
  state.overflowHighlightB = params->overflowHighlightB;
  state.backgroundR = params->backgroundColorR;
  state.backgroundG = params->backgroundColorG;
  state.backgroundB = params->backgroundColorB;
  state.fillVolume = params->identityOverlayEnabled;
  state.fillGrayRamp = params->identityOverlayRamp;
  state.fillResolution = params->identityOverlayRequestedSize;
  state.readGrayRamp = params->readGrayRamp;
  state.readIdentityPlot = params->readIdentityPlot;
  state.isolateIdentityData = params->isolateIdentityData;
  state.excludeIdentityData = params->excludeIdentityData;
  state.identityReadResolution = params->identityReadResolution;
  state.volumeSliceLassoRegion = params->volumeSlicingMode == "lasso";
  {
    const JsonField* explicitLasso =
        findField(object, "volumeSliceLassoRegion");
    if (explicitLasso &&
        !parseWireBool(*explicitLasso, &state.volumeSliceLassoRegion)) {
      if (error) *error = "invalid-volumeSliceLassoRegion";
      return false;
    }
  }
  state.volumeSliceRed = params->cubeSliceRed;
  state.volumeSliceYellow = params->cubeSliceYellow;
  state.volumeSliceGreen = params->cubeSliceGreen;
  state.volumeSliceCyan = params->cubeSliceCyan;
  state.volumeSliceBlue = params->cubeSliceBlue;
  state.volumeSliceMagenta = params->cubeSliceMagenta;
  state.neutralRadius = params->neutralRadius;
  state.chromaticityInputPrimaries = params->chromaticityInputPrimaries;
  state.chromaticityInputTransfer = params->chromaticityInputTransfer;
  state.chromaticityReferenceBasis = params->chromaticityReferenceBasis;
  state.chromaticityOverlayPrimaries = params->chromaticityOverlayPrimaries;
  state.chromaticityPlanckianLocus = params->chromaticityPlanckianLocus;
  state.chromaticitySpectralLocus3D = params->chromaticitySpectralLocus3D;
  state.glossNeighborhood = params->glossNeighborhood;
  state.glossLiftScale = params->glossLiftScale;
  state.glossSpatialInset = params->glossSpatialInset;
  state.glossBodyOpacity = params->glossBodyOpacity;
  state.glossHighlightOpacity = params->glossHighlightOpacity;
  state.glossPointCrispness = params->glossPointCrispness;
  state.glossHideText = params->glossHideText;
  state.waveformMode = params->viewerState.waveformMode;
  state.waveformHighDetail = params->viewerState.waveformHighDetail;
  state.waveformContinuousHighDetail =
      params->viewerState.waveformContinuousHighDetail;
  state.waveformHighDetailRequested =
      params->viewerState.waveformHighDetailRequested;
  state.waveformSampleColumns = params->viewerState.waveformSampleColumns;
  state.waveformSamplesPerColumn =
      params->viewerState.waveformSamplesPerColumn;
  state.waveformPointBrightness =
      params->viewerState.waveformPointBrightness;
  state.waveformGridBrightness = params->viewerState.waveformGridBrightness;
  state.waveformSaturation = params->viewerState.waveformSaturation;
  state.waveformDotSize = params->viewerState.waveformDotSize;
  state.waveformChannelRed = params->viewerState.waveformChannelRed;
  state.waveformChannelGreen = params->viewerState.waveformChannelGreen;
  state.waveformChannelBlue = params->viewerState.waveformChannelBlue;
  state.waveformChannelLuma = params->viewerState.waveformChannelLuma;
  state.waveformShowOverflow = params->viewerState.waveformShowOverflow;
  state.waveformHighlightOverflow =
      params->viewerState.waveformHighlightOverflow;
  state.waveformLumaMethod = params->viewerState.waveformLumaMethod;
  state.histogramMode = params->viewerState.histogramMode;
  state.histogramShowOverflow = params->viewerState.histogramShowOverflow;
  state.histogramHighlightOverflow =
      params->viewerState.histogramHighlightOverflow;
  state.scopeRangeMode = params->viewerState.scopeRangeMode;
  state.sourceDetailMode = params->viewerState.sourceDetailMode;
  state.sourceMaxProxyLongEdge = params->viewerState.sourceMaxProxyLongEdge;
  state.sourceUseNativeWhenAvailable =
      params->viewerState.sourceUseNativeWhenAvailable;
  state.sourceSyncSelections = params->viewerState.sourceSyncSelections;
  state.sourceSyncCommonPlotSettings =
      params->viewerState.sourceSyncCommonPlotSettings;
  state.sourceSessionId = params->senderId;
  state.sampleSettingsKey = params->cloudSettingsKey;
  params->viewerState = clampedViewerRuntimeState(std::move(state));

  params->pointSize = static_cast<float>(params->viewerState.pointSize);
  params->colorSaturation =
      static_cast<float>(params->viewerState.colorSaturation);
  params->glossNeighborhood = params->viewerState.glossNeighborhood;
  params->glossLiftScale =
      static_cast<float>(params->viewerState.glossLiftScale);
  params->glossBodyOpacity =
      static_cast<float>(params->viewerState.glossBodyOpacity);
  params->glossHighlightOpacity =
      static_cast<float>(params->viewerState.glossHighlightOpacity);
  params->glossPointCrispness =
      static_cast<float>(params->viewerState.glossPointCrispness);
  params->neutralRadius =
      static_cast<float>(params->viewerState.neutralRadius);
  params->overflowHighlightR =
      static_cast<float>(params->viewerState.overflowHighlightR);
  params->overflowHighlightG =
      static_cast<float>(params->viewerState.overflowHighlightG);
  params->overflowHighlightB =
      static_cast<float>(params->viewerState.overflowHighlightB);
  params->backgroundColorR =
      static_cast<float>(params->viewerState.backgroundR);
  params->backgroundColorG =
      static_cast<float>(params->viewerState.backgroundG);
  params->backgroundColorB =
      static_cast<float>(params->viewerState.backgroundB);
  params->sourceAspect =
      std::max(0.25f, std::min(4.0f, params->sourceAspect));
  params->pointDensity =
      std::max(0.1f, std::min(4.0f, params->pointDensity));
  params->identityOverlayRequestedSize =
      clampOverlaySize(params->identityOverlayRequestedSize);
  params->identityOverlaySize = clampOverlaySize(params->identityOverlaySize);
  params->identityReadResolution =
      clampOverlaySize(params->identityReadResolution);
  params->generatedIdentityResolution =
      params->generatedIdentityResolution > 0
          ? clampOverlaySize(params->generatedIdentityResolution)
          : 0;
  params->generatedIdentityStripBandCount =
      std::max(0, std::min(2, params->generatedIdentityStripBandCount));

#undef READ_STRING
#undef READ_INT
#undef READ_U64
#undef READ_FLOAT
#undef READ_BOOL
  return true;
}

ViewerLiveCommandKind kindForType(std::string_view type) {
  if (type == "params") return ViewerLiveCommandKind::Params;
  if (type == "clear_viewer_output") {
    return ViewerLiveCommandKind::ClearViewerOutput;
  }
  if (type == "heartbeat") return ViewerLiveCommandKind::Heartbeat;
  if (type == "bring_to_front") return ViewerLiveCommandKind::BringToFront;
  if (type == "disconnect") return ViewerLiveCommandKind::Disconnect;
  if (type == "shutdown") return ViewerLiveCommandKind::Shutdown;
  if (type == "input_cloud") return ViewerLiveCommandKind::InputCloud;
  if (type == "source_signal") return ViewerLiveCommandKind::SourceSignal;
  return ViewerLiveCommandKind::Unknown;
}

void incrementSaturated(std::size_t* value) noexcept {
  if (value && *value != std::numeric_limits<std::size_t>::max()) ++*value;
}

}  // namespace

ViewerLiveCommandDecodeResult decodeViewerLiveCommand(
    std::string_view line) noexcept {
  ViewerLiveCommandDecodeResult result{};
  try {
    if (line.empty()) {
      result.command.status = ViewerLiveCommandStatus::EmptyInput;
      result.command.reason = "empty-input";
      return result;
    }
    if (line.size() > kViewerLiveCommandMaxLineBytes) {
      result.command.status = ViewerLiveCommandStatus::Oversized;
      result.command.reason = "line-limit";
      return result;
    }
    ParsedObject object{};
    JsonCursor cursor(line);
    if (!cursor.parseTopObject(&object)) {
      result.command.status = cursor.status();
      result.command.reason = cursor.reason();
      return result;
    }
    std::string type;
    if (!requiredString(object, "type", &type, &result.command.reason)) {
      result.command.status = ViewerLiveCommandStatus::Invalid;
      return result;
    }
    result.command.kind = kindForType(type);
    if (result.command.kind == ViewerLiveCommandKind::Unknown) {
      result.command.status = ViewerLiveCommandStatus::UnknownType;
      result.command.reason = "unknown-type";
      return result;
    }
    if (result.command.kind == ViewerLiveCommandKind::Shutdown) {
      result.command.status = ViewerLiveCommandStatus::Accepted;
      return result;
    }
    if (!requiredSequence(object, &result.command.seq,
                          &result.command.reason) ||
        !requiredString(object, "senderId", &result.command.senderId,
                        &result.command.reason)) {
      result.command.status = ViewerLiveCommandStatus::Invalid;
      return result;
    }
    if (result.command.kind == ViewerLiveCommandKind::Params) {
      if (!decodeParams(object, &result.command.params,
                        &result.command.reason)) {
        result.command.status = ViewerLiveCommandStatus::Invalid;
        return result;
      }
      result.command.status = ViewerLiveCommandStatus::Accepted;
      return result;
    }
    if (result.command.kind == ViewerLiveCommandKind::ClearViewerOutput) {
      if (!optionalString(object, "reason", &result.command.reason,
                          &result.command.reason)) {
        result.command.status = ViewerLiveCommandStatus::Invalid;
        return result;
      }
    }
    if (result.command.kind == ViewerLiveCommandKind::InputCloud ||
        result.command.kind == ViewerLiveCommandKind::SourceSignal) {
      result.command.status = ViewerLiveCommandStatus::Dropped;
      result.command.reason = "resident-source-command-dropped";
      return result;
    }
    result.command.status = ViewerLiveCommandStatus::Accepted;
    return result;
  } catch (const std::bad_alloc&) {
    result.command = ViewerLiveCommand{};
    result.command.status = ViewerLiveCommandStatus::AllocationFailure;
    return result;
  } catch (...) {
    result.command = ViewerLiveCommand{};
    result.command.status = ViewerLiveCommandStatus::Invalid;
    return result;
  }
}

ViewerLiveCommandReducer::SenderWatermark*
ViewerLiveCommandReducer::findSenderLocked(std::string_view senderId) noexcept {
  for (auto& sender : senders_) {
    if (sender.used && sender.senderId == senderId) return &sender;
  }
  return nullptr;
}

const ViewerLiveCommandReducer::SenderWatermark*
ViewerLiveCommandReducer::findSenderLocked(
    std::string_view senderId) const noexcept {
  for (const auto& sender : senders_) {
    if (sender.used && sender.senderId == senderId) return &sender;
  }
  return nullptr;
}

ViewerLiveCommandReducer::SenderWatermark*
ViewerLiveCommandReducer::acquireSenderLocked(
    std::string_view senderId) noexcept {
  if (auto* existing = findSenderLocked(senderId)) return existing;
  for (auto& sender : senders_) {
    if (sender.used) continue;
    try {
      sender.senderId.assign(senderId.data(), senderId.size());
      sender.used = true;
      sender.lastParamsSequence = 0u;
      sender.lastClearSequence = 0u;
      return &sender;
    } catch (...) {
      sender.senderId.clear();
      sender.used = false;
      return nullptr;
    }
  }
  return nullptr;
}

ViewerLiveCommandSubmitResult ViewerLiveCommandReducer::submitLine(
    std::string_view line) noexcept {
  ViewerLiveCommandSubmitResult submitted{};
  const ViewerLiveCommandDecodeResult decoded = decodeViewerLiveCommand(line);
  submitted.kind = decoded.command.kind;
  submitted.status = decoded.command.status;
  submitted.seq = decoded.command.seq;
  try {
    submitted.senderId = decoded.command.senderId;
  } catch (...) {
    submitted.status = ViewerLiveCommandStatus::AllocationFailure;
    return submitted;
  }

  try {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!decoded.accepted()) {
      incrementSaturated(&pendingRejectedCount_);
      return submitted;
    }
    switch (decoded.command.kind) {
      case ViewerLiveCommandKind::Params: {
        SenderWatermark* sender = acquireSenderLocked(decoded.command.senderId);
        if (!sender) {
          submitted.status = ViewerLiveCommandStatus::SenderCapacityExceeded;
          incrementSaturated(&pendingRejectedCount_);
          return submitted;
        }
        if (decoded.command.seq <= sender->lastParamsSequence) {
          submitted.status = ViewerLiveCommandStatus::Stale;
          incrementSaturated(&pendingRejectedCount_);
          return submitted;
        }
        ViewerLiveCommandParams next = decoded.command.params;
        std::string nextSender = decoded.command.senderId;
        pendingParams_ = std::move(next);
        pendingSenderId_ = std::move(nextSender);
        hasPendingParams_ = true;
        sender->lastParamsSequence = decoded.command.seq;
        return submitted;
      }
      case ViewerLiveCommandKind::ClearViewerOutput: {
        const std::string& active =
            hasPendingParams_ ? pendingSenderId_ : committedSenderId_;
        if (active != decoded.command.senderId) {
          submitted.status = ViewerLiveCommandStatus::InactiveSender;
          incrementSaturated(&pendingRejectedCount_);
          return submitted;
        }
        SenderWatermark* sender = acquireSenderLocked(decoded.command.senderId);
        if (!sender) {
          submitted.status = ViewerLiveCommandStatus::SenderCapacityExceeded;
          incrementSaturated(&pendingRejectedCount_);
          return submitted;
        }
        if (decoded.command.seq <= sender->lastClearSequence) {
          submitted.status = ViewerLiveCommandStatus::Stale;
          incrementSaturated(&pendingRejectedCount_);
          return submitted;
        }
        std::string nextSender = decoded.command.senderId;
        std::string nextReason = decoded.command.reason;
        pendingClearSenderId_ = std::move(nextSender);
        pendingClearReason_ = std::move(nextReason);
        pendingClearSeq_ = decoded.command.seq;
        hasPendingClear_ = true;
        sender->lastClearSequence = decoded.command.seq;
        return submitted;
      }
      case ViewerLiveCommandKind::Heartbeat:
        pendingHeartbeatSenderId_ = decoded.command.senderId;
        pendingHeartbeat_ = true;
        return submitted;
      case ViewerLiveCommandKind::BringToFront:
        pendingBringToFront_ = true;
        return submitted;
      case ViewerLiveCommandKind::Disconnect:
        pendingDisconnect_ = true;
        return submitted;
      case ViewerLiveCommandKind::Shutdown:
        pendingShutdown_ = true;
        return submitted;
      case ViewerLiveCommandKind::InputCloud:
        incrementSaturated(&pendingDroppedInputCloudCount_);
        return submitted;
      case ViewerLiveCommandKind::SourceSignal:
        incrementSaturated(&pendingDroppedSourceSignalCount_);
        return submitted;
      default:
        incrementSaturated(&pendingRejectedCount_);
        return submitted;
    }
  } catch (const std::bad_alloc&) {
    submitted.status = ViewerLiveCommandStatus::AllocationFailure;
  } catch (...) {
    submitted.status = ViewerLiveCommandStatus::Invalid;
  }
  return submitted;
}

bool ViewerLiveCommandReducer::drain(ViewerLiveCommandBatch* output) noexcept {
  if (!output) return false;
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    ViewerLiveCommandBatch next{};
    next.previousSenderId = committedSenderId_;
    next.activeSenderId =
        hasPendingParams_ ? pendingSenderId_ : committedSenderId_;
    next.senderChanged = hasPendingParams_ &&
                         committedSenderId_ != pendingSenderId_;
    next.hasParams = hasPendingParams_;
    if (hasPendingParams_) next.params = pendingParams_;
    next.hasClear = hasPendingClear_;
    next.clearSeq = pendingClearSeq_;
    next.clearSenderId = pendingClearSenderId_;
    next.clearReason = pendingClearReason_;
    next.heartbeat = pendingHeartbeat_;
    next.heartbeatSenderId = pendingHeartbeatSenderId_;
    next.bringToFront = pendingBringToFront_;
    next.disconnected = pendingDisconnect_;
    next.shutdown = pendingShutdown_;
    next.droppedInputCloudCount = pendingDroppedInputCloudCount_;
    next.droppedSourceSignalCount = pendingDroppedSourceSignalCount_;
    next.rejectedCount = pendingRejectedCount_;
    *output = std::move(next);

    if (hasPendingParams_) committedSenderId_.swap(pendingSenderId_);
    hasPendingParams_ = false;
    pendingParams_ = ViewerLiveCommandParams{};
    pendingSenderId_.clear();
    hasPendingClear_ = false;
    pendingClearSeq_ = 0u;
    pendingClearSenderId_.clear();
    pendingClearReason_.clear();
    pendingHeartbeat_ = false;
    pendingHeartbeatSenderId_.clear();
    pendingBringToFront_ = false;
    pendingDisconnect_ = false;
    pendingShutdown_ = false;
    pendingDroppedInputCloudCount_ = 0u;
    pendingDroppedSourceSignalCount_ = 0u;
    pendingRejectedCount_ = 0u;
    return true;
  } catch (...) {
    return false;
  }
}

void ViewerLiveCommandReducer::reset() noexcept {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& sender : senders_) sender = SenderWatermark{};
    committedSenderId_.clear();
    pendingSenderId_.clear();
    pendingParams_ = ViewerLiveCommandParams{};
    hasPendingParams_ = false;
    pendingClearSeq_ = 0u;
    pendingClearSenderId_.clear();
    pendingClearReason_.clear();
    hasPendingClear_ = false;
    pendingHeartbeat_ = false;
    pendingHeartbeatSenderId_.clear();
    pendingBringToFront_ = false;
    pendingDisconnect_ = false;
    pendingShutdown_ = false;
    pendingDroppedInputCloudCount_ = 0u;
    pendingDroppedSourceSignalCount_ = 0u;
    pendingRejectedCount_ = 0u;
  } catch (...) {
  }
}

}  // namespace ChromaspaceViewer
