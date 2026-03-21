#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace WorkshopColor {

struct Vec2f {
  float x = 0.0f;
  float y = 0.0f;
};

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Mat3f {
  float m[3][3] = {{0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 0.0f}};
};

struct XyY {
  float x = 0.0f;
  float y = 0.0f;
  float Y = 0.0f;
};

enum class ColorPrimariesId : int {
  AcesAp0 = 0,
  AcesAp1,
  ArriWideGamut3,
  ArriWideGamut4,
  CanonCinemaGamut,
  DavinciWideGamut,
  FilmlightEGamut,
  FilmlightEGamut2,
  P3D65,
  Rec709,
  Rec2020,
  RedWideGamutRGB,
  SonySGamut3,
  SonySGamut3Cine,
  Xyz,
};

enum class TransferFunctionId : int {
  Linear = 0,
  SRgb,
  Gamma24,
  DavinciIntermediate,
  AcesCct,
  ArriLogC3,
  ArriLogC4,
  CanonLog2,
  SonySLog3,
  CanonLog3,
  RedLog3G10,
};

enum class ChromaticityReferenceBasis : int {
  CieStandardObserver = 0,
  InputObserver,
};

struct PrimariesDefinition {
  ColorPrimariesId id = ColorPrimariesId::Rec709;
  const char* key = "";
  const char* label = "";
  Vec2f red{};
  Vec2f green{};
  Vec2f blue{};
  Vec2f white{};
};

struct TransferFunctionDefinition {
  TransferFunctionId id = TransferFunctionId::Linear;
  const char* key = "";
  const char* label = "";
};

struct ChromaticityColorSpec {
  ColorPrimariesId inputPrimaries = ColorPrimariesId::DavinciWideGamut;
  TransferFunctionId inputTransfer = TransferFunctionId::DavinciIntermediate;
  ChromaticityReferenceBasis referenceBasis = ChromaticityReferenceBasis::CieStandardObserver;
  bool overlayEnabled = true;
  ColorPrimariesId overlayPrimaries = ColorPrimariesId::Rec709;
};

std::size_t primariesCount();
const PrimariesDefinition& primariesDefinition(std::size_t index);
const PrimariesDefinition& primariesDefinition(ColorPrimariesId id);
ColorPrimariesId primariesIdFromChoiceIndex(int index);
int primariesChoiceIndex(ColorPrimariesId id);

std::size_t transferFunctionCount();
const TransferFunctionDefinition& transferFunctionDefinition(std::size_t index);
const TransferFunctionDefinition& transferFunctionDefinition(TransferFunctionId id);
TransferFunctionId transferFunctionIdFromChoiceIndex(int index);
int transferFunctionChoiceIndex(TransferFunctionId id);

bool overlayPrimariesChoiceEnabled(int choiceIndex);
ColorPrimariesId overlayPrimariesIdFromChoiceIndex(int choiceIndex);
int overlayPrimariesChoiceIndex(bool enabled, ColorPrimariesId id);

Vec2f whitePoint(ColorPrimariesId id);
Mat3f rgbToXyzMatrix(ColorPrimariesId id);
Mat3f xyzToRgbMatrix(ColorPrimariesId id);
Vec3f mul(const Mat3f& matrix, Vec3f v);

Vec3f decodeToLinear(Vec3f rgb, TransferFunctionId tf);
Vec3f clamp(Vec3f rgb, float lo, float hi);
bool isFinite(Vec2f v);
bool isFinite(Vec3f v);

Vec3f xyToXyz(Vec2f xy, float Y = 1.0f);
XyY xyzToXyY(Vec3f xyz, Vec2f fallbackWhite);
Vec2f xyzToXy(Vec3f xyz, Vec2f fallbackWhite);

Vec2f standardObserverToInputObserver(Vec2f xy, ColorPrimariesId inputPrimaries);
Vec2f inputObserverToStandardObserver(Vec2f xy, ColorPrimariesId inputPrimaries);

const std::array<Vec3f, 82>& cie1931XyzCmfs5nm();
bool blackBodyChromaticity(float kelvin, Vec2f* xy);
float nearestBlackBodyTemperature(Vec2f xy, float minKelvin = 1000.0f, float maxKelvin = 20000.0f);
std::vector<Vec2f> blackBodyChromaticityCurve(float minKelvin, float maxKelvin, std::size_t steps);

}  // namespace WorkshopColor
