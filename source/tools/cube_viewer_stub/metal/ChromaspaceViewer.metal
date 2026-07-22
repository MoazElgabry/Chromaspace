#include <metal_stdlib>
using namespace metal;

constant float kTau = 6.28318530717958647692;
constant float kPi = 3.14159265358979323846;
constant uint kRasterPointCompactBlockWidth = 256u;

struct OverlayUniforms {
  int cubeSize;
  int ramp;
  int useInputPoints;
  int pointCount;
  float colorSaturation;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
};

struct InputUniforms {
  int pointCount;
  int inputStride;
  int glossView;
  float sourceAspect;
  float glossLiftScale;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct RasterSourceUniforms {
  InputUniforms input;
  int basePointCount;
  int sourceWidth;
  int sourceHeight;
  int sampleStride;
  int sampleCountX;
  int pixelFormat;
  int plotLinear;
  int plotLinearTransfer;
  int excludeIdentityData;
  int isolateIdentityData;
  int readIdentityPlot;
  int readGrayRamp;
  int identityCubeY1;
  int identityCubeY2;
  int identityRampY1;
  int identityRampY2;
  int identityCubeAppendOffset;
  int identityCubeAppendCount;
  int identityCubeAppendY1;
  int identityCubeAppendY2;
  int identityCubeAppendRowStep;
  int identityCubeAppendXStep;
  int identityRampAppendOffset;
  int identityRampAppendCount;
  int identityRampAppendY1;
  int identityRampAppendY2;
  int identityRampAppendRowStep;
  int identityRampAppendXStep;
  int occupancyFill;
  int occupancyAppendOffset;
  int occupancyAppendCount;
  int occupancyCandidateCount;
  int occupancyTargetThreshold;
  int lassoEnabled;
  int lassoStrokeCount;
  int lassoPointCount;
  int lassoStrokeFirst[16];
  int lassoStrokeCountPerStroke[16];
  int lassoStrokeSubtract[16];
  float lassoX[256];
  float lassoY[256];
  int cubeSlicingEnabled;
  int neutralRadiusEnabled;
  float neutralRadius;
  int cubeSliceRed;
  int cubeSliceYellow;
  int cubeSliceGreen;
  int cubeSliceCyan;
  int cubeSliceBlue;
  int cubeSliceMagenta;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct ScopeDensityUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int excludeOverflow;
  int onlyOverflow;
  int channelCount;
  int lumaMethod;
};

struct ScopeRangeUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  int includeOverflow;
  int lumaMethod;
  int previousRangeValid;
  float previousRangeMin;
  float previousRangeMax;
  int histogramBinCount;
};

struct HistogramSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
};

struct WaveformSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  float pointBrightness;
  float colorSaturation;
  float coverageAlpha;
};

struct GlossFieldAccumulateUniforms {
  int pointCount;
  int gridWidth;
  int gridHeight;
  int showOverflow;
};

struct GlossFieldCellUniforms {
  int cellCount;
  int gridWidth;
  int gridHeight;
  int neighborhoodChoice;
};

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

constant float kGlossFieldAccumScale = 1024.0;
constant float kGlossFieldAccumInvScale = 1.0 / 1024.0;

uint glossEncodeAccum(float v) {
  return uint(clamp(v, 0.0, 2.0) * kGlossFieldAccumScale + 0.5);
}

float glossDecodeAccum(uint v) {
  return float(v) * kGlossFieldAccumInvScale;
}

float glossCommonComponent(float r, float g, float b) {
  return max(0.0, min(r, min(g, b)));
}

float glossNeutrality(float r, float g, float b) {
  float common = glossCommonComponent(r, g, b);
  float maxRgb = max(r, max(g, b));
  return maxRgb > 1e-6 ? clamp(common / maxRgb, 0.0, 1.0) : 0.0;
}

float glossStrengthCue(float r, float g, float b) {
  float common = glossCommonComponent(r, g, b);
  float neutrality = glossNeutrality(r, g, b);
  return clamp(common * (0.75 + 0.85 * neutrality), 0.0, 1.0);
}

float glossPresenceWeight(float glossCue) {
  float t = clamp((glossCue - 0.06) / 0.22, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

float wrapHue01(float h) {
  h = fmod(h, 1.0);
  if (h < 0.0) h += 1.0;
  return h;
}

float luminanceAwareAlpha(float baseAlpha, float cr, float cg, float cb, float denseAlphaBias, bool overflowPoint,
                          float pointAlphaScale) {
  float alpha = baseAlpha * pointAlphaScale;
  if (overflowPoint || denseAlphaBias <= 0.0) {
    return clamp(alpha, 0.0, 1.0);
  }
  float luma = clamp(cr * 0.2126 + cg * 0.7152 + cb * 0.0722, 0.0, 1.0);
  float maxRgb = clamp(max(cr, max(cg, cb)), 0.0, 1.0);
  float value = mix(maxRgb, luma, 0.28);
  float highlightKnee = clamp((value - 0.70) / 0.24, 0.0, 1.0);
  float shadowMidProtect = 1.0 - clamp((value - 0.58) / 0.30, 0.0, 1.0);
  float multiplier = clamp(1.0 + 0.22 * denseAlphaBias * shadowMidProtect
                               - 0.12 * denseAlphaBias * highlightKnee,
                           0.94, 1.18);
  return clamp(alpha * multiplier, 0.0, 1.0);
}

float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6) return 0.0;
  float h = 0.0;
  if (cMax == r) {
    h = fmod((g - b) / delta, 6.0);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0;
  } else {
    h = ((r - g) / delta) + 4.0;
  }
  return wrapHue01(h / 6.0);
}

float safeDiv(float num, float den) {
  return fabs(den) < 1e-6 ? 0.0 : num / den;
}

float safeExp2Clamped(float value) {
  return exp2(clamp(value, -126.0, 126.0));
}

float safePowPos(float value, float exponent) {
  return value <= 0.0 ? 0.0 : pow(value, exponent);
}

float signPreservingPow(float value, float exponent) {
  return value == 0.0 ? 0.0 : copysign(safePowPos(fabs(value), exponent), value);
}

float exp10Compat(float value) {
  return safeExp2Clamped(value * 3.3219280948873626);
}

float decodeTransferChannel(float x, int tf) {
  switch (tf) {
    case 0: return x;
    case 1: {
      float a = fabs(x);
      float decoded = (a <= 0.04045) ? safeDiv(a, 12.92) : safePowPos(safeDiv(a + 0.055, 1.055), 2.4);
      return copysign(decoded, x);
    }
    case 2: return signPreservingPow(x, 2.4);
    case 3: return x <= 0.02740668 ? safeDiv(x, 10.44426855) : safeExp2Clamped(safeDiv(x, 0.07329248) - 7.0) - 0.0075;
    case 4: return x <= 0.155251141552511 ? safeDiv(x - 0.0729055341958355, 10.5402377416545) : safeExp2Clamped(x * 17.52 - 9.72);
    case 5: return x < 5.367655 * 0.010591 + 0.092809 ? safeDiv(x - 0.092809, 5.367655) : safeDiv(exp10Compat(safeDiv(x - 0.385537, 0.247190)) - 0.052272, 5.555556);
    case 6: return x < -0.7774983977293537 ? x * 0.3033266726886969 - 0.7774983977293537 : safeDiv(safeExp2Clamped(14.0 * safeDiv(x - 0.09286412512218964, 0.9071358748778103) + 6.0) - 64.0, 2231.8263090676883);
    case 7: {
      constexpr float kCut = 0.092864125;
      constexpr float kScale = 0.24136077;
      constexpr float kGain = 87.099375;
      float decoded = x < kCut ? -safeDiv(exp10Compat(safeDiv(kCut - x, kScale)) - 1.0, kGain) : safeDiv(exp10Compat(safeDiv(x - kCut, kScale)) - 1.0, kGain);
      return decoded * 0.9;
    }
    case 8: return x < 171.2102946929 / 1023.0 ? safeDiv((x * 1023.0 - 95.0) * 0.01125, 171.2102946929 - 95.0) : (exp10Compat(safeDiv(x * 1023.0 - 420.0, 261.5)) * 0.19 - 0.01);
    case 9:
      if (x < 0.04076162) return -safeDiv(exp10Compat(safeDiv(0.069886632 - x, 0.42889912)) - 1.0, 14.98325);
      if (x <= 0.105357102) return safeDiv(x - 0.073059361, 2.3069815);
      return safeDiv(exp10Compat(safeDiv(x - 0.073059361, 0.36726845)) - 1.0, 14.98325);
    case 10: return x < 0.0 ? safeDiv(x, 15.1927) - 0.01 : safeDiv(exp10Compat(safeDiv(x, 0.224282)) - 1.0, 155.975327) - 0.01;
    case 11: {
      constexpr float kA = 8.283605932402494;
      constexpr float kB = 0.09246575342465753;
      constexpr float kC = 0.5300133392291939;
      constexpr float kD = 0.08692876065491224;
      constexpr float kE = 0.005494072432257808;
      constexpr float kCut = kA * 0.005 + kB;
      return x < kCut ? safeDiv(x - kB, kA) : exp(safeDiv(x - kC, kD)) - kE;
    }
    case 12: return x <= 0.14 ? safeDiv(x - 0.0929, 6.025) : safeDiv(exp10Compat(3.89616 * x - 2.27752) - 0.0108, 0.9892);
    case 13: {
      constexpr float kA = 0.555556;
      constexpr float kB = 0.009468;
      constexpr float kC = 0.344676;
      constexpr float kD = 0.790453;
      constexpr float kE = 8.735631;
      constexpr float kF = 0.092864;
      constexpr float kCut = 0.100537775223865;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 14: {
      constexpr float kA = 5.555556;
      constexpr float kB = 0.064829;
      constexpr float kC = 0.245281;
      constexpr float kD = 0.384316;
      constexpr float kE = 8.799461;
      constexpr float kF = 0.092864;
      constexpr float kCut = 0.100686685370811;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 15: return x < 0.181 ? safeDiv(x - 0.125, 5.6) : exp10Compat(safeDiv(x - 0.598206, 0.241514)) - 0.00873;
    case 16: return signPreservingPow(x, 2.2);
    case 17: return signPreservingPow(x, 2.6);
    default: return x;
  }
}

float3 mulRows(constant float* m, float3 v) {
  return float3(dot(float3(m[0], m[1], m[2]), v),
                dot(float3(m[3], m[4], m[5]), v),
                dot(float3(m[6], m[7], m[8]), v));
}

float3 xyToXyz(float2 xy, float Y) {
  if (fabs(xy.y) <= 1e-8) return float3(xy.x, Y, 1.0 - xy.x);
  return float3(xy.x * Y / xy.y, Y, (1.0 - xy.x - xy.y) * Y / xy.y);
}

float3 xyzToXyY(float3 xyz, float2 fallbackWhite) {
  if (fabs(xyz.y) <= 1e-8) return float3(fallbackWhite.x, fallbackWhite.y, 0.0);
  float sum = xyz.x + xyz.y + xyz.z;
  if (fabs(sum) <= 1e-8) return float3(fallbackWhite.x, fallbackWhite.y, xyz.y);
  return float3(xyz.x / sum, xyz.y / sum, xyz.y);
}

float3 mapChromaticityPosition(float r, float g, float b, constant InputUniforms& u) {
  float3 linear = float3(decodeTransferChannel(r, u.chromaticityInputTransfer),
                         decodeTransferChannel(g, u.chromaticityInputTransfer),
                         decodeTransferChannel(b, u.chromaticityInputTransfer));
  if (u.showOverflow == 0) linear = clamp(linear, 0.0, 1.0);
  float3 xyz = mulRows(u.chromaticityRgbToXyz, linear);
  float2 white = float2(u.chromaticityWhiteX, u.chromaticityWhiteY);
  float3 xyY = xyzToXyY(xyz, white);
  float2 xy = xyY.xy;
  if (u.chromaticityReferenceBasis != 0) {
    float3 basisXyz = xyToXyz(xy, 1.0);
    float3 rgb = mulRows(u.chromaticityXyzToRgb, basisXyz);
    xy = xyzToXyY(rgb, float2(1.0 / 3.0, 1.0 / 3.0)).xy;
  }
  float viewerHeight = ((u.showOverflow != 0) ? xyY.z : clamp(xyY.z, 0.0, 1.0)) * 2.0 - 1.0;
  return float3((xy.x - (1.0 / 3.0)) * 3.0,
                (xy.y - (1.0 / 3.0)) * 3.0,
                viewerHeight);
}

float3 mapChromaticityPosition(float r, float g, float b, constant OverlayUniforms& u) {
  float3 linear = float3(decodeTransferChannel(r, u.chromaticityInputTransfer),
                         decodeTransferChannel(g, u.chromaticityInputTransfer),
                         decodeTransferChannel(b, u.chromaticityInputTransfer));
  linear = clamp(linear, 0.0, 1.0);
  float3 xyz = mulRows(u.chromaticityRgbToXyz, linear);
  float2 white = float2(u.chromaticityWhiteX, u.chromaticityWhiteY);
  float3 xyY = xyzToXyY(xyz, white);
  float2 xy = xyY.xy;
  if (u.chromaticityReferenceBasis != 0) {
    float3 basisXyz = xyToXyz(xy, 1.0);
    float3 rgb = mulRows(u.chromaticityXyzToRgb, basisXyz);
    xy = xyzToXyY(rgb, float2(1.0 / 3.0, 1.0 / 3.0)).xy;
  }
  float viewerHeight = clamp(xyY.z, 0.0, 1.0) * 2.0 - 1.0;
  return float3((xy.x - (1.0 / 3.0)) * 3.0,
                (xy.y - (1.0 / 3.0)) * 3.0,
                viewerHeight);
}

float2 rgbToHsvHexconePlane(float r, float g, float b) {
  return float2(r - 0.5 * g - 0.5 * b, 0.8660254037844386 * (g - b));
}

float3 mapPlotPosition(float r, float g, float b, int plotMode, int circularHsl, int circularHsv, int normConeNormalized, int showOverflow) {
  if (plotMode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    float delta = cMax - cMin;
    float l = 0.5 * (cMax + cMin);
    float h = rawRgbHue01(r, g, b, cMax, delta);
    float satDenom = 1.0 - abs(2.0 * l - 1.0);
    if (delta > 1e-6 && satDenom < 0.0) {
      h = wrapHue01(h + 0.5);
    }
    float angle = h * kTau;
    float radius = delta;
    if (circularHsl != 0) {
      float denom = satDenom;
      if (abs(denom) <= 1e-6) {
        denom = (denom < 0.0) ? -1e-6 : 1e-6;
      }
      radius = abs(delta / denom);
    }
    return float3(cos(angle) * radius, l * 2.0 - 1.0, sin(angle) * radius);
  }
  if (plotMode == 2) {
    float cMax = max(r, max(g, b));
    if (circularHsv != 0) {
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      float h = rawRgbHue01(r, g, b, cMax, delta);
      float sat = (delta > 1e-6 && cMax > 1e-6) ? (delta / cMax) : 0.0;
      float angle = h * kTau;
      return float3(cos(angle) * sat, cMax * 2.0 - 1.0, sin(angle) * sat);
    }
    float2 plane = rgbToHsvHexconePlane(r, g, b);
    return float3(plane.x, cMax * 2.0 - 1.0, plane.y);
  }
  if (plotMode == 3) {
    float rotX = r * 0.81649658 + g * -0.40824829 + b * -0.40824829;
    float rotY = g * 0.70710678 + b * -0.70710678;
    float rotZ = r * 0.57735027 + g * 0.57735027 + b * 0.57735027;
    float azimuth = atan2(rotY, rotX);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float wrappedHue = azimuth < 0.0 ? azimuth + kTau : azimuth;
    float polar = atan2(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float c = polar * 1.0467733744265997;
    float l = radius3 * 0.5773502691896258;
    float polarScaled = c * 0.9553166181245093;
    float radial = l * sin(polarScaled) / 0.816496580927726;
    float angle = wrappedHue;
    return float3(cos(angle) * radial, l * 2.0 - 1.0, sin(angle) * radial);
  }
  if (plotMode == 4 || plotMode == 5) {
    bool jpOverflow = (showOverflow != 0 && plotMode == 5);
    float rr = jpOverflow ? r : clamp01(r);
    float gg = jpOverflow ? g : clamp01(g);
    float bb = jpOverflow ? b : clamp01(b);
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan2(rotY, rotX);
    if (hue < 0.0) hue += kTau;
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float magnitude = 0.0;
    if (plotMode == 4) {
      magnitude = clamp(radius3 * 0.576, 0.0, 1.0);
    } else {
      float kAsinInvSqrt2 = asin(1.0 / sqrt(2.0));
      float kAsinInvSqrt3 = asin(1.0 / sqrt(3.0));
      float kHueCoef1 = 1.0 / (2.0 - (kAsinInvSqrt2 / kAsinInvSqrt3));
      float huecoef2 = 2.0 * polar * sin((2.0 * kPi / 3.0) - fmod(hue, kPi / 3.0)) / sqrt(3.0);
      float huemag = ((acos(cos(3.0 * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0)) * huecoef2;
      float satmag = sin(huemag + kAsinInvSqrt3);
      magnitude = radius3 * satmag;
      if (jpOverflow && magnitude < 0.0) {
        magnitude = -magnitude;
        hue += kPi;
        if (hue >= kTau) hue -= kTau;
      }
      magnitude = jpOverflow ? magnitude : clamp(magnitude, 0.0, 1.0);
    }
    float phiNorm = jpOverflow ? max(polar / 0.9553166181245093, 0.0) : clamp(polar / 0.9553166181245093, 0.0, 1.0);
    float phi = phiNorm * 0.9553166181245093;
    float radial = magnitude * sin(phi);
    return float3(cos(hue) * radial, magnitude * cos(phi) * 2.0 - 1.0, sin(hue) * radial);
  }
  if (plotMode == 6) {
    bool normConeOverflow = (showOverflow != 0 && plotMode == 6);
    float rr = normConeOverflow ? r : clamp(r, 0.0, 1.0);
    float gg = normConeOverflow ? g : clamp(g, 0.0, 1.0);
    float bb = normConeOverflow ? b : clamp(b, 0.0, 1.0);
    float maxRgb = max(rr, max(gg, bb));
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan2(rotY, rotX) / kTau;
    if (hue < 0.0) hue += 1.0;
    float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
    float polar = atan2(chromaRadius, rotZ);
    float chroma = polar / 0.9553166181245093;
    if (normConeNormalized != 0) {
      float angle = hue * kTau - kPi / 6.0;
      float cosPolar = cos(polar);
      float safeCos = abs(cosPolar) > 1e-6 ? cosPolar : (cosPolar < 0.0 ? -1e-6 : 1e-6);
      float cone = (sin(polar) / safeCos) / sqrt(2.0);
      float sinTerm = clamp(sin(3.0 * angle), -1.0, 1.0);
      float chromaGain = 1.0 / (2.0 * cos(acos(sinTerm) / 3.0));
      chroma = chromaGain > 1e-6 ? cone / chromaGain : 0.0;
      if (normConeOverflow && chroma < 0.0) {
        chroma = -chroma;
        hue += 0.5;
        if (hue >= 1.0) hue -= 1.0;
      }
    }
    chroma = normConeOverflow ? max(chroma, 0.0) : clamp(chroma, 0.0, 1.0);
    float value = normConeOverflow ? maxRgb : clamp(maxRgb, 0.0, 1.0);
    float angle = hue * kTau;
    return float3(cos(angle) * chroma, value * 2.0 - 1.0, sin(angle) * chroma);
  }
  if (plotMode == 7) {
    bool reuleauxOverflow = (showOverflow != 0 && plotMode == 7);
    float rr = reuleauxOverflow ? r : clamp01(r);
    float gg = reuleauxOverflow ? g : clamp01(g);
    float bb = reuleauxOverflow ? b : clamp01(b);
    float rotX = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float rotY = (gg - bb) * 0.40824829046;
    float rotZ = (rr + gg + bb) / 3.0;
    float hue = kPi - atan2(rotY, -rotX);
    if (hue < 0.0) hue += kTau;
    if (hue >= kTau) hue = fmod(hue, kTau);
    float sat = abs(rotZ) <= 1e-6 ? 0.0 : length(float2(rotX, rotY)) / rotZ;
    if (reuleauxOverflow && sat < 0.0) {
      sat = -sat;
      hue += kPi;
      if (hue >= kTau) hue -= kTau;
    }
    sat = reuleauxOverflow ? sat / 1.41421356237 : clamp(sat / 1.41421356237, 0.0, 1.0);
    float value = reuleauxOverflow ? max(rr, max(gg, bb)) : clamp(max(rr, max(gg, bb)), 0.0, 1.0);
    return float3(cos(hue) * sat, value * 2.0 - 1.0, sin(hue) * sat);
  }
  return float3(r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0);
}

void mapDisplayColor(float inR, float inG, float inB, thread float& outR, thread float& outG, thread float& outB) {
  outR = pow(clamp01(inR), 1.0 / 2.2);
  outG = pow(clamp01(inG), 1.0 / 2.2);
  outB = pow(clamp01(inB), 1.0 / 2.2);
}

void rgbToHsl(float r, float g, float b, thread float& outH, thread float& outS, thread float& outL) {
  float cMax = max(r, max(g, b));
  float cMin = min(r, min(g, b));
  float delta = cMax - cMin;
  outL = 0.5 * (cMax + cMin);
  outH = 0.0;
  outS = 0.0;
  if (delta > 1e-6) {
    float denom = max(1e-6, 1.0 - abs(2.0 * outL - 1.0));
    outS = delta / denom;
    outH = rawRgbHue01(r, g, b, cMax, delta);
  }
}

float hueToRgbChannel(float p, float q, float t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
  if (t < 1.0 / 2.0) return q;
  if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  return p;
}

void hslToRgb(float h, float s, float l, thread float& outR, thread float& outG, thread float& outB) {
  h = wrapHue01(h);
  s = clamp01(s);
  l = clamp01(l);
  if (s <= 1e-6) {
    outR = l;
    outG = l;
    outB = l;
    return;
  }
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  outR = clamp01(hueToRgbChannel(p, q, h + 1.0 / 3.0));
  outG = clamp01(hueToRgbChannel(p, q, h));
  outB = clamp01(hueToRgbChannel(p, q, h - 1.0 / 3.0));
}

void applyDisplaySaturation(float saturation, thread float& r, thread float& g, thread float& b) {
  float sat = clamp(saturation, 1.0, 6.0);
  float baseR = clamp01(r);
  float baseG = clamp01(g);
  float baseB = clamp01(b);
  float luma = clamp(baseR * 0.2126 + baseG * 0.7152 + baseB * 0.0722, 0.0, 1.0);
  if (sat <= 1.0) {
    r = max(0.0, luma + (baseR - luma) * sat);
    g = max(0.0, luma + (baseG - luma) * sat);
    b = max(0.0, luma + (baseB - luma) * sat);
  } else {
    float h = 0.0;
    float s = 0.0;
    float l = 0.0;
    rgbToHsl(baseR, baseG, baseB, h, s, l);
    if (s <= 1e-5) {
      r = baseR;
      g = baseG;
      b = baseB;
    } else {
      float t = clamp((sat - 1.0) / 5.0, 0.0, 1.0);
      float shaped = pow(t, 0.55);
      float targetS = clamp(s + (1.0 - s) * (0.32 + 0.68 * shaped), 0.0, 1.0);
      float highlight = clamp((l - 0.58) / 0.34, 0.0, 1.0);
      float targetL = clamp(l - highlight * (0.08 + 0.10 * shaped), 0.0, 1.0);
      float boostedR = baseR;
      float boostedG = baseG;
      float boostedB = baseB;
      hslToRgb(h, targetS, targetL, boostedR, boostedG, boostedB);
      float mixAmount = clamp(0.24 + 0.76 * shaped, 0.0, 1.0);
      r = max(0.0, baseR * (1.0 - mixAmount) + boostedR * mixAmount);
      g = max(0.0, baseG * (1.0 - mixAmount) + boostedG * mixAmount);
      b = max(0.0, baseB * (1.0 - mixAmount) + boostedB * mixAmount);
    }
  }
  float peak = max(r, max(g, b));
  if (peak > 1.0) {
    r /= peak;
    g /= peak;
    b /= peak;
  }
  r = clamp(r, 0.0, 1.0);
  g = clamp(g, 0.0, 1.0);
  b = clamp(b, 0.0, 1.0);
}

bool pointOverflowsCube(float r, float g, float b) {
  return r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0;
}

void writeHiddenInputPoint(device packed_float3* vertVals, device float4* colorVals, uint index) {
  vertVals[index] = packed_float3(0.0, 0.0, 0.0);
  colorVals[index] = float4(0.0, 0.0, 0.0, 0.0);
}

void writeMappedInputPoint(device packed_float3* vertVals,
                           device float4* colorVals,
                           uint index,
                           float xNorm,
                           float yNorm,
                           float r,
                           float g,
                           float b,
                           constant InputUniforms& u) {
  bool overflowPoint = pointOverflowsCube(r, g, b);
  float plotR = (u.showOverflow != 0) ? r : clamp01(r);
  float plotG = (u.showOverflow != 0) ? g : clamp01(g);
  float plotB = (u.showOverflow != 0) ? b : clamp01(b);
  float3 pos = mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow);
  if (u.plotMode == 8) {
    pos = mapChromaticityPosition(r, g, b, u);
  }
  if (u.glossView != 0) {
    float aspect = clamp(u.sourceAspect, 0.25, 4.0);
    float halfWidth = aspect >= 1.0 ? 1.22 : (1.22 * aspect);
    float halfDepth = aspect >= 1.0 ? (1.22 / aspect) : 1.22;
    float common = glossCommonComponent(plotR, plotG, plotB);
    float bodyR = max(plotR - common, 0.0);
    float bodyG = max(plotG - common, 0.0);
    float bodyB = max(plotB - common, 0.0);
    float bodyLuma = clamp(bodyR * 0.2126 + bodyG * 0.7152 + bodyB * 0.0722, 0.0, 1.0);
    float glossCue = glossStrengthCue(plotR, plotG, plotB);
    float glossPresence = glossPresenceWeight(glossCue);
    float xPos = -halfWidth + (2.0 * halfWidth * xNorm);
    float zPos = halfDepth - (2.0 * halfDepth * yNorm);
    float yPos = -0.92 + bodyLuma * 0.92 + glossCue * glossPresence * u.glossLiftScale * 1.34;
    pos = float3(xPos, yPos, zPos);
  }
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(r, g, b, cr, cg, cb);
    applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
    if (u.glossView != 0) {
      float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
      float neutralBlend = clamp(0.08 + 0.52 * glossPresence, 0.0, 0.62);
      float brightnessGain = 1.18 + 1.20 * glossPresence;
      cr = clamp((cr * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cg = clamp((cg * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cb = clamp((cb * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
    }
  }
  bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint);
  float baseAlpha = overflowHighlighted ? 0.95 : 0.72;
  if (u.glossView != 0 && !overflowHighlighted) {
    float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
    baseAlpha = 0.01 + 0.97 * glossPresence;
  }
  colorVals[index] = float4(cr, cg, cb,
                            luminanceAwareAlpha(baseAlpha,
                                                cr,
                                                cg,
                                                cb,
                                                u.denseAlphaBias,
                                                overflowHighlighted,
                                                u.pointAlphaScale));
}

bool rasterSourceRowInRange(int y, int y1, int y2) {
  return y1 >= 0 && y2 > y1 && y >= y1 && y < y2;
}

bool rasterSourceRowInCube(constant RasterSourceUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityCubeY1, u.identityCubeY2);
}

bool rasterSourceRowInRamp(constant RasterSourceUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityRampY1, u.identityRampY2);
}

bool rasterAppendSampleCoords(uint index,
                              int offset,
                              int count,
                              int y1,
                              int y2,
                              int rowStep,
                              int xStep,
                              int sourceWidth,
                              thread int* outX,
                              thread int* outY) {
  if (count <= 0 || index < uint(max(offset, 0)) || index >= uint(max(offset + count, offset))) {
    return false;
  }
  int local = int(index) - offset;
  int safeXStep = max(xStep, 1);
  int safeRowStep = max(rowStep, 1);
  int samplesPerRow = max(1, (max(sourceWidth, 0) + safeXStep - 1) / safeXStep);
  int rowIndex = local / samplesPerRow;
  int xIndex = local - rowIndex * samplesPerRow;
  *outX = min(max(xIndex * safeXStep, 0), max(sourceWidth - 1, 0));
  *outY = min(max(y1 + rowIndex * safeRowStep, y1), max(y2 - 1, y1));
  return true;
}

float rasterHalton(uint index, uint base) {
  float f = 1.0;
  float r = 0.0;
  while (index > 0u) {
    f /= float(base);
    r += f * float(index % base);
    index /= base;
  }
  return r;
}

bool rasterOccupancySampleCoords(uint index, constant RasterSourceUniforms& u, thread int* outX, thread int* outY) {
  if (u.occupancyAppendCount <= 0 || index < uint(max(u.occupancyAppendOffset, 0)) ||
      index >= uint(max(u.occupancyAppendOffset + u.occupancyAppendCount, u.occupancyAppendOffset))) {
    return false;
  }
  uint local = index - uint(max(u.occupancyAppendOffset, 0));
  uint attemptCount = uint(max(u.occupancyCandidateCount, u.occupancyAppendCount));
  uint attempt = attemptCount > 0u ? (local * max(attemptCount, 1u)) / uint(max(u.occupancyAppendCount, 1)) : local;
  float xNorm = rasterHalton(attempt + 1u, 2u);
  float yNorm = rasterHalton(attempt + 1u, 3u);
  *outX = min(max(int(xNorm * float(max(u.sourceWidth, 1))), 0), max(u.sourceWidth - 1, 0));
  *outY = min(max(int(yNorm * float(max(u.sourceHeight, 1))), 0), max(u.sourceHeight - 1, 0));
  return true;
}

bool rasterLassoPointInStroke(constant RasterSourceUniforms& u, int strokeIndex, float xNorm, float yNorm) {
  if (strokeIndex < 0 || strokeIndex >= min(max(u.lassoStrokeCount, 0), 16)) return false;
  int first = u.lassoStrokeFirst[strokeIndex];
  int count = u.lassoStrokeCountPerStroke[strokeIndex];
  if (count < 3 || first < 0 || first + count > min(max(u.lassoPointCount, 0), 256)) return false;
  bool inside = false;
  for (int i = 0, j = count - 1; i < count; j = i++) {
    float xi = u.lassoX[first + i];
    float yi = u.lassoY[first + i];
    float xj = u.lassoX[first + j];
    float yj = u.lassoY[first + j];
    bool intersects = ((yi > yNorm) != (yj > yNorm)) &&
                      (xNorm < (xj - xi) * (yNorm - yi) / ((yj - yi) + 1.0e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

bool rasterLassoContainsPoint(constant RasterSourceUniforms& u, float xNorm, float yNorm) {
  bool inside = false;
  int strokeCount = min(max(u.lassoStrokeCount, 0), 16);
  for (int stroke = 0; stroke < strokeCount; ++stroke) {
    if (!rasterLassoPointInStroke(u, stroke, xNorm, yNorm)) continue;
    inside = u.lassoStrokeSubtract[stroke] == 0;
  }
  return inside;
}

float rasterNeutralRadius(float r, float g, float b, constant RasterSourceUniforms& u) {
  const float kRgbAxisMaxRadius = 0.8164965809277260;
  const float kPolarMax = 0.9553166181245093;
  const float kChenPolarScale = 1.0467733744265997;
  int mode = u.input.plotMode;
  if (mode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    if (u.input.circularHsl != 0) {
      float l = 0.5 * (cMax + cMin);
      float denom = 1.0 - abs(2.0 * l - 1.0);
      if (abs(denom) <= 1e-6) denom = denom < 0.0 ? -1e-6 : 1e-6;
      return clamp01(abs((cMax - cMin) / denom));
    }
    return clamp01(cMax - cMin);
  }
  if (mode == 2) {
    if (u.input.circularHsv != 0) {
      float cMax = max(r, max(g, b));
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      return (delta > 1e-6 && cMax > 1e-6) ? clamp01(delta / cMax) : 0.0;
    }
    float x = r - 0.5 * g - 0.5 * b;
    float z = 0.8660254037844386 * (g - b);
    return clamp01(sqrt(x * x + z * z));
  }
  bool overflowMode = u.input.showOverflow != 0 && (mode == 5 || mode == 6 || mode == 7);
  float rr = overflowMode ? r : clamp01(r);
  float gg = overflowMode ? g : clamp01(g);
  float bb = overflowMode ? b : clamp01(b);
  float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
  float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
  float rotZ = 0.57735026919 * (rr + gg + bb);
  float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
  if (mode == 3) {
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(chromaRadius, max(rotZ, 1e-8));
    float light = radius3 * 0.5773502691896258;
    float radial = light * sin(polar * kChenPolarScale) / kRgbAxisMaxRadius;
    return clamp01(radial);
  }
  if (mode == 4 || mode == 5) {
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(chromaRadius, rotZ);
    float radial = radius3 * sin((polar / kPolarMax) * kPolarMax);
    return clamp01(radial / sin(kPolarMax));
  }
  if (mode == 6) {
    float polar = atan2(chromaRadius, rotZ);
    return clamp01(polar / kPolarMax);
  }
  if (mode == 7) {
    float rotZAvg = (rr + gg + bb) / 3.0;
    float rx = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float ry = (gg - bb) * 0.40824829046;
    float sat = abs(rotZAvg) <= 1e-6 ? 0.0 : sqrt(rx * rx + ry * ry) / rotZAvg;
    return clamp01(abs(sat) / 1.41421356237);
  }
  return clamp01(sqrt(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius);
}

bool rasterCubeSliceContains(float r, float g, float b, constant RasterSourceUniforms& u) {
  if (u.neutralRadiusEnabled != 0 && u.input.plotMode != 8 && u.input.showOverflow == 0) {
    float threshold = clamp01(u.neutralRadius) * clamp01(u.neutralRadius);
    if (rasterNeutralRadius(r, g, b, u) > threshold + 1.0e-6) return false;
  }
  if (u.cubeSlicingEnabled == 0) return true;
  bool anySelected = u.cubeSliceRed || u.cubeSliceYellow || u.cubeSliceGreen ||
                     u.cubeSliceCyan || u.cubeSliceBlue || u.cubeSliceMagenta;
  if (!anySelected) return false;
  if (u.input.plotMode == 0 || u.input.glossView != 0) {
    const float kEps = 1.0e-6;
    bool geRG = r + kEps >= g;
    bool geGB = g + kEps >= b;
    bool geGR = g + kEps >= r;
    bool geRB = r + kEps >= b;
    bool geBG = b + kEps >= g;
    bool geBR = b + kEps >= r;
    if (u.cubeSliceRed && geRG && geGB) return true;
    if (u.cubeSliceYellow && geGR && geRB) return true;
    if (u.cubeSliceGreen && geGB && geBR) return true;
    if (u.cubeSliceCyan && geBG && geGR) return true;
    if (u.cubeSliceBlue && geBR && geRG) return true;
    if (u.cubeSliceMagenta && geRB && geBG) return true;
    return false;
  }
  float cMax = max(r, max(g, b));
  float cMin = min(r, min(g, b));
  float delta = cMax - cMin;
  if (delta <= 1.0e-6) return false;
  float hue = wrapHue01(rawRgbHue01(r, g, b, cMax, delta));
  int sector = int(floor((hue + (1.0 / 12.0)) * 6.0)) % 6;
  if (sector == 0) return u.cubeSliceRed != 0;
  if (sector == 1) return u.cubeSliceYellow != 0;
  if (sector == 2) return u.cubeSliceGreen != 0;
  if (sector == 3) return u.cubeSliceCyan != 0;
  if (sector == 4) return u.cubeSliceBlue != 0;
  return u.cubeSliceMagenta != 0;
}

void readRasterSourceRgb(const device half4* source16,
                         const device float4* source32,
                         constant RasterSourceUniforms& u,
                         int x,
                         int y,
                         thread float& r,
                         thread float& g,
                         thread float& b) {
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  uint pixel = uint(y * u.sourceWidth + x);
  if (u.pixelFormat == 1) {
    float4 v = source32[pixel];
    r = v.x;
    g = v.y;
    b = v.z;
  } else {
    half4 v = source16[pixel];
    r = float(v.x);
    g = float(v.y);
    b = float(v.z);
  }
}

int rasterOccupancyComponentBin(float value) {
  if (value < 0.0) return 0;
  if (value > 1.0) return 17;
  return 1 + min(max(int(floor(value * 16.0)), 0), 15);
}

int rasterOccupancyBinIndex(float r, float g, float b) {
  return (rasterOccupancyComponentBin(r) * 18 + rasterOccupancyComponentBin(g)) * 18 +
         rasterOccupancyComponentBin(b);
}

bool rasterSampleVisible(constant RasterSourceUniforms& u,
                         int x,
                         int y,
                         float xNorm,
                         float yNorm,
                         float r,
                         float g,
                         float b) {
  bool inCubeStrip = rasterSourceRowInCube(u, y);
  bool inRampStrip = rasterSourceRowInRamp(u, y);
  bool inAnyIdentityStrip = inCubeStrip || inRampStrip;
  bool visible = true;
  if (u.excludeIdentityData != 0 && inAnyIdentityStrip) {
    visible = false;
  } else if (u.isolateIdentityData != 0) {
    visible = (u.readIdentityPlot != 0 && inCubeStrip) || (u.readGrayRamp != 0 && inRampStrip);
  }
  if (visible && u.lassoEnabled != 0 && !rasterLassoContainsPoint(u, xNorm, yNorm)) {
    visible = false;
  }
  if (visible && !rasterCubeSliceContains(r, g, b, u)) {
    visible = false;
  }
  (void)x;
  return visible;
}

void rasterReadTransformedSample(const device half4* source16,
                                 const device float4* source32,
                                 constant RasterSourceUniforms& u,
                                 int x,
                                 int y,
                                 thread float& r,
                                 thread float& g,
                                 thread float& b) {
  readRasterSourceRgb(source16, source32, u, x, y, r, g, b);
  if (u.plotLinear != 0 && u.input.plotMode != 8) {
    r = decodeTransferChannel(r, u.plotLinearTransfer);
    g = decodeTransferChannel(g, u.plotLinearTransfer);
    b = decodeTransferChannel(b, u.plotLinearTransfer);
  }
}

void rasterReadTransformedSampleTexture(texture2d<float, access::read> sourceTexture,
                                        constant RasterSourceUniforms& u,
                                        int x,
                                        int y,
                                        thread float& r,
                                        thread float& g,
                                        thread float& b) {
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float4 v = sourceTexture.read(uint2(uint(x), uint(y)));
  r = v.x;
  g = v.y;
  b = v.z;
  if (u.plotLinear != 0 && u.input.plotMode != 8) {
    r = decodeTransferChannel(r, u.plotLinearTransfer);
    g = decodeTransferChannel(g, u.plotLinearTransfer);
    b = decodeTransferChannel(b, u.plotLinearTransfer);
  }
}

kernel void rasterOccupancyCountKernel(const device half4* source16 [[buffer(0)]],
                                       const device float4* source32 [[buffer(1)]],
                                       device atomic_uint* occupancyBins [[buffer(2)]],
                                       device atomic_uint* visibleCount [[buffer(3)]],
                                       constant RasterSourceUniforms& u [[buffer(4)]],
                                       uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.basePointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSample(source16, source32, u, x, y, r, g, b);
  if (!rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b)) return;
  atomic_fetch_add_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&visibleCount[0], 1u, memory_order_relaxed);
}

kernel void rasterSourceKernel(const device half4* source16 [[buffer(0)]],
                               const device float4* source32 [[buffer(1)]],
                               device packed_float3* vertVals [[buffer(2)]],
                               device float4* colorVals [[buffer(3)]],
                               device atomic_uint* occupancyBins [[buffer(4)]],
                               constant RasterSourceUniforms& u [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.input.pointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  bool occupancyCandidate = false;
  if (!rasterOccupancySampleCoords(index, u, &x, &y)) {
    if (!rasterAppendSampleCoords(index,
                                  u.identityCubeAppendOffset,
                                  u.identityCubeAppendCount,
                                  u.identityCubeAppendY1,
                                  u.identityCubeAppendY2,
                                  u.identityCubeAppendRowStep,
                                  u.identityCubeAppendXStep,
                                  u.sourceWidth,
                                  &x,
                                  &y)) {
      rasterAppendSampleCoords(index,
                               u.identityRampAppendOffset,
                               u.identityRampAppendCount,
                               u.identityRampAppendY1,
                               u.identityRampAppendY2,
                               u.identityRampAppendRowStep,
                               u.identityRampAppendXStep,
                               u.sourceWidth,
                               &x,
                               &y);
    }
  } else {
    occupancyCandidate = true;
  }
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSample(source16, source32, u, x, y, r, g, b);
  bool visible = rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b);
  if (visible && occupancyCandidate) {
    uint binCount = atomic_load_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], memory_order_relaxed);
    visible = int(binCount) <= max(u.occupancyTargetThreshold, 0);
  }
  if (!visible) {
    writeHiddenInputPoint(vertVals, colorVals, index);
    return;
  }
  writeMappedInputPoint(vertVals, colorVals, index, xNorm, yNorm, r, g, b, u.input);
}

kernel void rasterOccupancyTextureCountKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                              device atomic_uint* occupancyBins [[buffer(0)]],
                                              device atomic_uint* visibleCount [[buffer(1)]],
                                              constant RasterSourceUniforms& u [[buffer(2)]],
                                              uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.basePointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSampleTexture(sourceTexture, u, x, y, r, g, b);
  if (!rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b)) return;
  atomic_fetch_add_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&visibleCount[0], 1u, memory_order_relaxed);
}

kernel void rasterOccupancyThresholdKernel(const device atomic_uint* visibleCount [[buffer(0)]],
                                           device RasterSourceUniforms* uniforms [[buffer(1)]],
                                           uint index [[thread_position_in_grid]]) {
  if (index != 0u) return;
  constexpr float kOccupancyBinCount = 5832.0;
  uint count = atomic_load_explicit(&visibleCount[0], memory_order_relaxed);
  float meanOccupancy = float(count) / kOccupancyBinCount;
  uniforms[0].occupancyTargetThreshold = max(0, int(ceil(meanOccupancy * 0.72)));
}

kernel void rasterSourceTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                      device packed_float3* vertVals [[buffer(0)]],
                                      device float4* colorVals [[buffer(1)]],
                                      device atomic_uint* occupancyBins [[buffer(2)]],
                                      constant RasterSourceUniforms& u [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.input.pointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  bool occupancyCandidate = false;
  if (!rasterOccupancySampleCoords(index, u, &x, &y)) {
    if (!rasterAppendSampleCoords(index,
                                  u.identityCubeAppendOffset,
                                  u.identityCubeAppendCount,
                                  u.identityCubeAppendY1,
                                  u.identityCubeAppendY2,
                                  u.identityCubeAppendRowStep,
                                  u.identityCubeAppendXStep,
                                  u.sourceWidth,
                                  &x,
                                  &y)) {
      rasterAppendSampleCoords(index,
                               u.identityRampAppendOffset,
                               u.identityRampAppendCount,
                               u.identityRampAppendY1,
                               u.identityRampAppendY2,
                               u.identityRampAppendRowStep,
                               u.identityRampAppendXStep,
                               u.sourceWidth,
                               &x,
                               &y);
    }
  } else {
    occupancyCandidate = true;
  }
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSampleTexture(sourceTexture, u, x, y, r, g, b);
  bool visible = rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b);
  if (visible && occupancyCandidate) {
    uint binCount = atomic_load_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], memory_order_relaxed);
    visible = int(binCount) <= max(u.occupancyTargetThreshold, 0);
  }
  if (!visible) {
    writeHiddenInputPoint(vertVals, colorVals, index);
    return;
  }
  writeMappedInputPoint(vertVals, colorVals, index, xNorm, yNorm, r, g, b, u.input);
}

// Stable device-side compaction uses a fixed 256-thread Blelloch scan per
// block.  Every lane participates in the scan, including lanes past the
// logical point count (those lanes contribute zero).
kernel void rasterPointCompactLocalScanKernel(
    const device float4* colors [[buffer(0)]],
    device uint* localOffsets [[buffer(1)]],
    device uint* blockSums [[buffer(2)]],
    constant uint& pointCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint scan[kRasterPointCompactBlockWidth];

  uint visible = 0u;
  if (gid < pointCount && colors[gid].a > 0.0f) {
    visible = 1u;
  }
  scan[lane] = visible;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Upsweep: build a reduction tree in shared memory.
  for (uint offset = 1u; offset < kRasterPointCompactBlockWidth; offset <<= 1u) {
    uint index = ((lane + 1u) * (offset << 1u)) - 1u;
    if (index < kRasterPointCompactBlockWidth) {
      scan[index] += scan[index - offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint blockTotal = scan[kRasterPointCompactBlockWidth - 1u];
  if (lane == 0u) {
    scan[kRasterPointCompactBlockWidth - 1u] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Downsweep: convert the reduction tree to exclusive offsets.
  for (uint offset = kRasterPointCompactBlockWidth >> 1u; offset > 0u;
       offset >>= 1u) {
    uint index = ((lane + 1u) * (offset << 1u)) - 1u;
    if (index < kRasterPointCompactBlockWidth) {
      uint left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (gid < pointCount) {
    localOffsets[gid] = scan[lane];
  }
  if (lane == 0u) {
    blockSums[gid / kRasterPointCompactBlockWidth] = blockTotal;
  }
}

// Scan one level of the block-sum hierarchy.  The interface intentionally
// matches the first-level scan so callers can recurse until one block remains.
kernel void rasterPointScanBlockSumsKernel(
    const device uint* counts [[buffer(0)]],
    device uint* localOffsets [[buffer(1)]],
    device uint* nextBlockSums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint scan[kRasterPointCompactBlockWidth];

  uint value = 0u;
  if (gid < count) {
    value = counts[gid];
  }
  scan[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = 1u; offset < kRasterPointCompactBlockWidth; offset <<= 1u) {
    uint index = ((lane + 1u) * (offset << 1u)) - 1u;
    if (index < kRasterPointCompactBlockWidth) {
      scan[index] += scan[index - offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint blockTotal = scan[kRasterPointCompactBlockWidth - 1u];
  if (lane == 0u) {
    scan[kRasterPointCompactBlockWidth - 1u] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = kRasterPointCompactBlockWidth >> 1u; offset > 0u;
       offset >>= 1u) {
    uint index = ((lane + 1u) * (offset << 1u)) - 1u;
    if (index < kRasterPointCompactBlockWidth) {
      uint left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (gid < count) {
    localOffsets[gid] = scan[lane];
  }
  if (lane == 0u) {
    nextBlockSums[gid / kRasterPointCompactBlockWidth] = blockTotal;
  }
}

kernel void rasterPointAddBlockOffsetsKernel(
    device uint* localOffsets [[buffer(0)]],
    const device uint* parentOffsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  localOffsets[gid] += parentOffsets[gid / kRasterPointCompactBlockWidth];
}

kernel void rasterPointCompactScatterKernel(
    const device packed_float3* inputPositions [[buffer(0)]],
    const device float4* inputColors [[buffer(1)]],
    const device uint* pointLocalOffsets [[buffer(2)]],
    const device uint* firstLevelBlockOffsets [[buffer(3)]],
    device packed_float3* outputPositions [[buffer(4)]],
    device float4* outputColors [[buffer(5)]],
    constant uint& pointCount [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= pointCount) {
    return;
  }
  const float4 color = inputColors[gid];
  if (!(color.a > 0.0f)) {
    return;
  }
  const uint destination =
      firstLevelBlockOffsets[gid / kRasterPointCompactBlockWidth] +
      pointLocalOffsets[gid];
  outputPositions[destination] = inputPositions[gid];
  outputColors[destination] = color;
}

kernel void rasterPointFinalizeIndirectArgsKernel(
    const device uint* firstLevelBlockSums [[buffer(0)]],
    const device uint* firstLevelBlockOffsets [[buffer(1)]],
    constant uint& blockCount [[buffer(2)]],
    device uint* indirectArgs [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) {
    return;
  }
  uint vertexCount = 0u;
  if (blockCount > 0u) {
    const uint lastBlock = blockCount - 1u;
    vertexCount = firstLevelBlockOffsets[lastBlock] +
                  firstLevelBlockSums[lastBlock];
  }
  indirectArgs[0] = vertexCount;
  indirectArgs[1] = 1u;
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}

kernel void overlayKernel(const device float4* inputVals [[buffer(0)]],
                          device packed_float3* vertVals [[buffer(1)]],
                          device float4* colorVals [[buffer(2)]],
                          constant OverlayUniforms& u [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  uint cubeSize = uint(max(u.cubeSize, 1));
  uint cubePoints = cubeSize * cubeSize * cubeSize;
  uint rampPoints = (u.ramp != 0) ? (cubeSize * cubeSize) : 0u;
  uint uploadedPoints = uint(max(u.pointCount, 0));
  uint total = (u.useInputPoints != 0) ? uploadedPoints : (cubePoints + rampPoints);
  if (index >= total) return;

  float r;
  float g;
  float b;
  float alpha;
  if (u.useInputPoints != 0) {
    float4 p = inputVals[index];
    r = p.x;
    g = p.y;
    b = p.z;
    alpha = p.w;
  } else if (index < cubePoints) {
    uint denom = max(cubeSize - 1u, 1u);
    uint rx = index % cubeSize;
    uint gy = (index / cubeSize) % cubeSize;
    uint bz = index / (cubeSize * cubeSize);
    r = float(rx) / float(denom);
    g = float(gy) / float(denom);
    b = float(bz) / float(denom);
    alpha = 0.24;
  } else {
    uint rampIndex = index - cubePoints;
    uint rampCount = max(rampPoints, 1u);
    float t = float(rampIndex) / float(max(rampCount - 1u, 1u));
    r = t;
    g = t;
    b = t;
    alpha = 0.92;
  }

  float3 pos = u.plotMode == 8
                   ? mapChromaticityPosition(r, g, b, u)
                   : mapPlotPosition(r, g, b, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, 0);
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  mapDisplayColor(r, g, b, cr, cg, cb);
  applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
  colorVals[index] = float4(cr, cg, cb, alpha);
}

kernel void inputKernel(const device float* inputVals [[buffer(0)]],
                        device packed_float3* vertVals [[buffer(1)]],
                        device float4* colorVals [[buffer(2)]],
                        constant InputUniforms& u [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total) return;
  uint stride = uint(max(u.inputStride, 3));
  uint base = index * stride;
  float xNorm = 0.5;
  float yNorm = 0.5;
  float r = inputVals[base + 0];
  float g = inputVals[base + 1];
  float b = inputVals[base + 2];
  if (u.glossView != 0 && stride >= 6) {
    xNorm = clamp(inputVals[base + 0], 0.0, 1.0);
    yNorm = clamp(inputVals[base + 1], 0.0, 1.0);
    r = inputVals[base + 3];
    g = inputVals[base + 4];
    b = inputVals[base + 5];
  }
  bool overflowPoint = pointOverflowsCube(r, g, b);
  float plotR = (u.showOverflow != 0) ? r : clamp01(r);
  float plotG = (u.showOverflow != 0) ? g : clamp01(g);
  float plotB = (u.showOverflow != 0) ? b : clamp01(b);
  float3 pos = mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow);
  if (u.plotMode == 8) {
    pos = mapChromaticityPosition(r, g, b, u);
  }
  if (u.glossView != 0) {
    float aspect = clamp(u.sourceAspect, 0.25, 4.0);
    float halfWidth = aspect >= 1.0 ? 1.22 : (1.22 * aspect);
    float halfDepth = aspect >= 1.0 ? (1.22 / aspect) : 1.22;
    float common = glossCommonComponent(plotR, plotG, plotB);
    float bodyR = max(plotR - common, 0.0);
    float bodyG = max(plotG - common, 0.0);
    float bodyB = max(plotB - common, 0.0);
    float bodyLuma = clamp(bodyR * 0.2126 + bodyG * 0.7152 + bodyB * 0.0722, 0.0, 1.0);
    float glossCue = glossStrengthCue(plotR, plotG, plotB);
    float glossPresence = glossPresenceWeight(glossCue);
    float xPos = -halfWidth + (2.0 * halfWidth * xNorm);
    float zPos = halfDepth - (2.0 * halfDepth * yNorm);
    float yPos = -0.92 + bodyLuma * 0.92 + glossCue * glossPresence * u.glossLiftScale * 1.34;
    pos = float3(xPos, yPos, zPos);
  }
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(r, g, b, cr, cg, cb);
    applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
    if (u.glossView != 0) {
      float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
      float neutralBlend = clamp(0.08 + 0.52 * glossPresence, 0.0, 0.62);
      float brightnessGain = 1.18 + 1.20 * glossPresence;
      cr = clamp((cr * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cg = clamp((cg * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cb = clamp((cb * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
    }
  }
  bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint);
  float baseAlpha = overflowHighlighted ? 0.95 : 0.72;
  if (u.glossView != 0 && !overflowHighlighted) {
    float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
    baseAlpha = 0.01 + 0.97 * glossPresence;
  }
  colorVals[index] = float4(cr, cg, cb,
                            luminanceAwareAlpha(baseAlpha,
                                                cr,
                                                cg,
                                                cb,
                                                u.denseAlphaBias,
                                                overflowHighlighted,
                                                u.pointAlphaScale));
}

kernel void inputSampleKernel(const device packed_float3* srcVerts [[buffer(0)]],
                              const device float4* srcColors [[buffer(1)]],
                              device packed_float3* dstVerts [[buffer(2)]],
                              device float4* dstColors [[buffer(3)]],
                              constant InputSampleUniforms& u [[buffer(4)]],
                              uint index [[thread_position_in_grid]]) {
  uint visible = uint(max(u.visiblePointCount, 0));
  uint full = uint(max(u.fullPointCount, 0));
  if (index >= visible) return;
  uint srcIndex = 0u;
  if (visible > 1u && full > 1u) {
    float t = float(index) / float(visible - 1u);
    srcIndex = min(uint(floor(t * float(full - 1u) + 0.5)), full - 1u);
  }
  packed_float3 src = srcVerts[srcIndex];
  dstVerts[index] = packed_float3(src.x, src.y, src.z);
  dstColors[index] = srcColors[srcIndex];
}

void accumulateScopeDensity(device atomic_uint* density,
                            constant ScopeDensityUniforms& u,
                            int channel,
                            float xNorm,
                            float value) {
  int channelCount = max(u.channelCount, 1);
  if (channel < 0 || channel >= channelCount) return;
  bool overflow = (value < 0.0 || value > 1.0);
  if (u.onlyOverflow != 0 && !overflow) return;
  if (u.excludeOverflow != 0 && (value < 0.0 || value > 1.0)) return;
  int x = clamp(int(xNorm * float(u.width)), 0, max(u.width - 1, 0));
  int signalBins = u.waveform != 0 ? u.height : u.width;
  int y = clamp(int((value - u.rangeMin) * u.invRange * float(signalBins)),
                0,
                max(signalBins - 1, 0));
  int binIndex = u.waveform != 0
      ? (channel * u.width + x) * u.height + y
      : channel * u.width + y;
  atomic_fetch_add_explicit(&density[binIndex], 1u, memory_order_relaxed);
}

float scopeLuma(float r, float g, float b, int method) {
  switch (method) {
    case 1:
      return 0.2627 * r + 0.6780 * g + 0.0593 * b;
    case 2:
      return 0.2990 * r + 0.5870 * g + 0.1140 * b;
    case 3:
      return (r + g + b) / 3.0;
    default:
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
}

uint orderedUintFromFloat(float value) {
  uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

float floatFromOrderedUint(uint value) {
  uint bits = (value & 0x80000000u) != 0u ? (value ^ 0x80000000u) : ~value;
  return as_type<float>(bits);
}

void includeScopeRangeValue(float value,
                            constant ScopeRangeUniforms& range,
                            device atomic_uint* rangeBits) {
  if (rangeBits == nullptr) return;
  if (range.includeOverflow == 0 && (value < 0.0 || value > 1.0)) return;
  uint ordered = orderedUintFromFloat(value);
  atomic_fetch_min_explicit(&rangeBits[0], ordered, memory_order_relaxed);
  atomic_fetch_max_explicit(&rangeBits[1], ordered, memory_order_relaxed);
  atomic_fetch_add_explicit(&rangeBits[2], 1u, memory_order_relaxed);
}

void includeScopeRangeValues(float r,
                             float g,
                             float b,
                             constant ScopeRangeUniforms& range,
                             device atomic_uint* rangeBits) {
  bool lumaOnly = (range.waveform != 0 && range.scopeMode == 2) ||
                  (range.waveform == 0 && range.scopeMode == 1);
  if (lumaOnly) {
    includeScopeRangeValue(scopeLuma(r, g, b, range.lumaMethod), range, rangeBits);
    return;
  }
  if (range.includeRed != 0) includeScopeRangeValue(r, range, rangeBits);
  if (range.includeGreen != 0) includeScopeRangeValue(g, range, rangeBits);
  if (range.includeBlue != 0) includeScopeRangeValue(b, range, rangeBits);
  if (range.waveform != 0 && range.scopeMode == 1 && range.includeLuma != 0) {
    includeScopeRangeValue(scopeLuma(r, g, b, range.lumaMethod), range, rangeBits);
  }
}

void includeScopeRangeHistogramValue(float value,
                                     constant ScopeRangeUniforms& range,
                                     float minValue,
                                     float invRange,
                                     device atomic_uint* histogram) {
  if (histogram == nullptr || range.histogramBinCount <= 0) return;
  if (range.includeOverflow == 0 && (value < 0.0 || value > 1.0)) return;
  int bin = clamp(int(floor((value - minValue) * invRange * float(range.histogramBinCount))),
                  0,
                  max(range.histogramBinCount - 1, 0));
  atomic_fetch_add_explicit(&histogram[uint(bin)], 1u, memory_order_relaxed);
}

void includeScopeRangeHistogramValues(float r,
                                      float g,
                                      float b,
                                      constant ScopeRangeUniforms& range,
                                      float minValue,
                                      float invRange,
                                      device atomic_uint* histogram) {
  bool lumaOnly = (range.waveform != 0 && range.scopeMode == 2) ||
                  (range.waveform == 0 && range.scopeMode == 1);
  if (lumaOnly) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogram);
    return;
  }
  if (range.includeRed != 0) {
    includeScopeRangeHistogramValue(r, range, minValue, invRange, histogram);
  }
  if (range.includeGreen != 0) {
    includeScopeRangeHistogramValue(g, range, minValue, invRange, histogram);
  }
  if (range.includeBlue != 0) {
    includeScopeRangeHistogramValue(b, range, minValue, invRange, histogram);
  }
  if (range.waveform != 0 && range.scopeMode == 1 && range.includeLuma != 0) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogram);
  }
}

bool rasterScopeSampleFromTexture(texture2d<float, access::read> sourceTexture,
                                  constant RasterSourceUniforms& raster,
                                  uint index,
                                  thread float* outXNorm,
                                  thread float* outYNorm,
                                  thread float* outR,
                                  thread float* outG,
                                  thread float* outB) {
  int sampleCountX = max(raster.sampleCountX, 1);
  int stride = max(raster.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  bool haveCoords = index < uint(max(raster.basePointCount, 0));
  if (!haveCoords) {
    haveCoords = rasterAppendSampleCoords(index,
                                          raster.identityCubeAppendOffset,
                                          raster.identityCubeAppendCount,
                                          raster.identityCubeAppendY1,
                                          raster.identityCubeAppendY2,
                                          raster.identityCubeAppendRowStep,
                                          raster.identityCubeAppendXStep,
                                          raster.sourceWidth,
                                          &x,
                                          &y);
    if (!haveCoords) {
      haveCoords = rasterAppendSampleCoords(index,
                                            raster.identityRampAppendOffset,
                                            raster.identityRampAppendCount,
                                            raster.identityRampAppendY1,
                                            raster.identityRampAppendY2,
                                            raster.identityRampAppendRowStep,
                                            raster.identityRampAppendXStep,
                                            raster.sourceWidth,
                                            &x,
                                            &y);
    }
  }
  if (!haveCoords) return false;
  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(raster.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(raster.sourceHeight, 1));
  if (outXNorm != nullptr) *outXNorm = xNorm;
  if (outYNorm != nullptr) *outYNorm = yNorm;
  rasterReadTransformedSampleTexture(sourceTexture, raster, x, y, *outR, *outG, *outB);
  return rasterSampleVisible(raster, x, y, xNorm, yNorm, *outR, *outG, *outB);
}

kernel void scopeDensityKernel(const device float* samples [[buffer(0)]],
                               device atomic_uint* density [[buffer(1)]],
                               constant ScopeDensityUniforms& u [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total || u.width <= 0 || u.height <= 0) return;
  uint base = index * 5u;
  float xNorm = clamp(samples[base + 0u], 0.0, 1.0);
  float r = samples[base + 2u];
  float g = samples[base + 3u];
  float b = samples[base + 4u];
  bool lumaOnly = (u.waveform != 0 && u.scopeMode == 2) ||
                  (u.waveform == 0 && u.scopeMode == 1);
  if (lumaOnly) {
    accumulateScopeDensity(density, u, 0, xNorm, scopeLuma(r, g, b, u.lumaMethod));
  } else {
    accumulateScopeDensity(density, u, 0, xNorm, r);
    accumulateScopeDensity(density, u, 1, xNorm, g);
    accumulateScopeDensity(density, u, 2, xNorm, b);
    if (u.waveform != 0 && u.scopeMode == 1 && u.channelCount >= 4) {
      accumulateScopeDensity(density, u, 3, xNorm, scopeLuma(r, g, b, u.lumaMethod));
    }
  }
}

kernel void rasterScopeDensityTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                            device atomic_uint* density [[buffer(0)]],
                                            constant RasterSourceUniforms& raster [[buffer(1)]],
                                            constant ScopeDensityUniforms& scope [[buffer(2)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || scope.width <= 0 || scope.height <= 0 || density == nullptr) return;
  float xNorm = 0.5;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, &xNorm, nullptr, &r, &g, &b)) return;

  bool lumaOnly = (scope.waveform != 0 && scope.scopeMode == 2) ||
                  (scope.waveform == 0 && scope.scopeMode == 1);
  if (lumaOnly) {
    accumulateScopeDensity(density, scope, 0, xNorm, scopeLuma(r, g, b, scope.lumaMethod));
  } else {
    accumulateScopeDensity(density, scope, 0, xNorm, r);
    accumulateScopeDensity(density, scope, 1, xNorm, g);
    accumulateScopeDensity(density, scope, 2, xNorm, b);
    if (scope.waveform != 0 && scope.scopeMode == 1 && scope.channelCount >= 4) {
      accumulateScopeDensity(density, scope, 3, xNorm, scopeLuma(r, g, b, scope.lumaMethod));
    }
  }
}

kernel void rasterScopeRangeTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                          device atomic_uint* rangeBits [[buffer(0)]],
                                          constant RasterSourceUniforms& raster [[buffer(1)]],
                                          constant ScopeRangeUniforms& range [[buffer(2)]],
                                          uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || rangeBits == nullptr) return;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, nullptr, nullptr, &r, &g, &b)) return;
  includeScopeRangeValues(r, g, b, range, rangeBits);
}

kernel void rasterScopeRangeHistogramTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                                   device atomic_uint* histogram [[buffer(0)]],
                                                   const device atomic_uint* rangeBits [[buffer(1)]],
                                                   constant RasterSourceUniforms& raster [[buffer(2)]],
                                                   constant ScopeRangeUniforms& range [[buffer(3)]],
                                                   uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || histogram == nullptr || rangeBits == nullptr) return;
  uint validCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (validCount == 0u) return;
  float minValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[0], memory_order_relaxed));
  float maxValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[1], memory_order_relaxed));
  float invRange = 1.0 / max(1.0e-7, maxValue - minValue);
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, nullptr, nullptr, &r, &g, &b)) return;
  includeScopeRangeHistogramValues(r, g, b, range, minValue, invRange, histogram);
}

kernel void rasterGlossFieldAccumulateTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                                    device atomic_uint* occupancyCounts [[buffer(0)]],
                                                    device atomic_uint* sumR [[buffer(1)]],
                                                    device atomic_uint* sumG [[buffer(2)]],
                                                    device atomic_uint* sumB [[buffer(3)]],
                                                    device atomic_uint* sumY [[buffer(4)]],
                                                    device atomic_uint* sumMax [[buffer(5)]],
                                                    device atomic_uint* sumMin [[buffer(6)]],
                                                    device atomic_uint* sumNeutrality [[buffer(7)]],
                                                    constant RasterSourceUniforms& raster [[buffer(8)]],
                                                    constant GlossFieldAccumulateUniforms& u [[buffer(9)]],
                                                    uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total) return;
  float xNorm = 0.5;
  float yNorm = 0.5;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, &xNorm, &yNorm, &r, &g, &b)) return;
  if (u.showOverflow == 0) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  float maxRgb = clamp(max(r, max(g, b)), 0.0, 1.0);
  float minRgb = clamp(max(0.0, min(r, min(g, b))), 0.0, 1.0);
  float neutralityValue = maxRgb > 1e-6 ? clamp(minRgb / maxRgb, 0.0, 1.0) : 0.0;
  float luma = clamp(r * 0.2126 + g * 0.7152 + b * 0.0722, 0.0, 1.0);
  int x = clamp(int(xNorm * float(u.gridWidth)), 0, max(u.gridWidth - 1, 0));
  int y = clamp(int((1.0 - yNorm) * float(u.gridHeight)), 0, max(u.gridHeight - 1, 0));
  uint cellIndex = uint(y * u.gridWidth + x);
  atomic_fetch_add_explicit(&occupancyCounts[cellIndex], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&sumR[cellIndex], glossEncodeAccum(r), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumG[cellIndex], glossEncodeAccum(g), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumB[cellIndex], glossEncodeAccum(b), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumY[cellIndex], glossEncodeAccum(luma), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMax[cellIndex], glossEncodeAccum(maxRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMin[cellIndex], glossEncodeAccum(minRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumNeutrality[cellIndex], glossEncodeAccum(neutralityValue), memory_order_relaxed);
}

kernel void scopeRangeHistogramPercentileKernel(const device atomic_uint* histogram [[buffer(0)]],
                                                const device atomic_uint* rangeBits [[buffer(1)]],
                                                device atomic_uint* percentileBits [[buffer(2)]],
                                                constant ScopeRangeUniforms& range [[buffer(3)]],
                                                uint index [[thread_position_in_grid]]) {
  if (index != 0u || histogram == nullptr || rangeBits == nullptr ||
      percentileBits == nullptr || range.histogramBinCount <= 0) return;
  uint totalCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (totalCount == 0u) {
    atomic_store_explicit(&percentileBits[0], orderedUintFromFloat(0.0), memory_order_relaxed);
    atomic_store_explicit(&percentileBits[1], orderedUintFromFloat(1.0), memory_order_relaxed);
    return;
  }
  float minValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[0], memory_order_relaxed));
  float maxValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[1], memory_order_relaxed));
  ulong lowTarget = ulong(totalCount) / 1000ul;
  ulong highTarget = (ulong(totalCount) * 999ul) / 1000ul;
  ulong accumulated = 0ul;
  float lowValue = minValue;
  float highValue = maxValue;
  bool foundLow = false;
  bool foundHigh = false;
  float span = maxValue - minValue;
  for (int bin = 0; bin < range.histogramBinCount; ++bin) {
    accumulated += ulong(atomic_load_explicit(&histogram[uint(bin)], memory_order_relaxed));
    if (!foundLow && accumulated > lowTarget) {
      float t = (float(bin) + 0.5) / float(range.histogramBinCount);
      lowValue = minValue + t * span;
      foundLow = true;
    }
    if (!foundHigh && accumulated > highTarget) {
      float t = (float(bin) + 0.5) / float(range.histogramBinCount);
      highValue = minValue + t * span;
      foundHigh = true;
      break;
    }
  }
  atomic_store_explicit(&percentileBits[0], orderedUintFromFloat(lowValue), memory_order_relaxed);
  atomic_store_explicit(&percentileBits[1], orderedUintFromFloat(highValue), memory_order_relaxed);
}

kernel void scopeRangeFinalizeKernel(const device atomic_uint* percentileBits [[buffer(0)]],
                                     const device atomic_uint* rangeBits [[buffer(1)]],
                                     device atomic_uint* finalRangeBits [[buffer(2)]],
                                     constant ScopeRangeUniforms& range [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
  if (index != 0u || percentileBits == nullptr || rangeBits == nullptr || finalRangeBits == nullptr) return;
  uint validCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (validCount == 0u) {
    atomic_store_explicit(&finalRangeBits[0], orderedUintFromFloat(0.0), memory_order_relaxed);
    atomic_store_explicit(&finalRangeBits[1], orderedUintFromFloat(1.0), memory_order_relaxed);
    atomic_store_explicit(&finalRangeBits[2], 0u, memory_order_relaxed);
    return;
  }
  float rangeMin = min(0.0, floatFromOrderedUint(atomic_load_explicit(&percentileBits[0], memory_order_relaxed)));
  float rangeMax = max(1.0, floatFromOrderedUint(atomic_load_explicit(&percentileBits[1], memory_order_relaxed)));
  float pad = max(0.02, (rangeMax - rangeMin) * 0.04);
  rangeMin -= pad;
  rangeMax += pad;
  if (!(rangeMax > rangeMin + 1.0e-5)) {
    rangeMin = 0.0;
    rangeMax = 1.0;
  }
  if (range.previousRangeValid != 0 && range.previousRangeMax > range.previousRangeMin + 1.0e-5) {
    rangeMin = rangeMin < range.previousRangeMin
                   ? rangeMin
                   : range.previousRangeMin + (rangeMin - range.previousRangeMin) * 0.16;
    rangeMax = rangeMax > range.previousRangeMax
                   ? rangeMax
                   : range.previousRangeMax + (rangeMax - range.previousRangeMax) * 0.16;
  }
  atomic_store_explicit(&finalRangeBits[0], orderedUintFromFloat(rangeMin), memory_order_relaxed);
  atomic_store_explicit(&finalRangeBits[1], orderedUintFromFloat(rangeMax), memory_order_relaxed);
  atomic_store_explicit(&finalRangeBits[2], validCount, memory_order_relaxed);
}

void resolvedScopeRange(const device atomic_uint* finalRangeBits,
                        thread float* outRangeMin,
                        thread float* outInvRange) {
  float rangeMin = 0.0;
  float rangeMax = 1.0;
  if (finalRangeBits != nullptr &&
      atomic_load_explicit(&finalRangeBits[2], memory_order_relaxed) != 0u) {
    rangeMin =
        floatFromOrderedUint(atomic_load_explicit(&finalRangeBits[0], memory_order_relaxed));
    rangeMax =
        floatFromOrderedUint(atomic_load_explicit(&finalRangeBits[1], memory_order_relaxed));
  }
  if (!(rangeMax > rangeMin + 1.0e-5)) {
    rangeMin = 0.0;
    rangeMax = 1.0;
  }
  *outRangeMin = rangeMin;
  *outInvRange = 1.0 / max(1.0e-6, rangeMax - rangeMin);
}

kernel void histogramSurfaceApplyRangeKernel(
    const device atomic_uint* finalRangeBits [[buffer(0)]],
    device ScopeDensityUniforms* density [[buffer(1)]],
    device ScopeDensityUniforms* overflow [[buffer(2)]],
    device HistogramSurfaceUniforms* surface [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0u || density == nullptr || overflow == nullptr || surface == nullptr) return;
  float rangeMin = 0.0;
  float invRange = 1.0;
  resolvedScopeRange(finalRangeBits, &rangeMin, &invRange);
  density[0].rangeMin = rangeMin;
  density[0].invRange = invRange;
  overflow[0].rangeMin = rangeMin;
  overflow[0].invRange = invRange;
  surface[0].rangeMin = rangeMin;
  surface[0].invRange = invRange;
}

kernel void waveformSurfaceApplyRangeKernel(
    const device atomic_uint* finalRangeBits [[buffer(0)]],
    device ScopeDensityUniforms* density [[buffer(1)]],
    device ScopeDensityUniforms* overflow [[buffer(2)]],
    device WaveformSurfaceUniforms* surface [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0u || density == nullptr || overflow == nullptr || surface == nullptr) return;
  float rangeMin = 0.0;
  float invRange = 1.0;
  resolvedScopeRange(finalRangeBits, &rangeMin, &invRange);
  density[0].rangeMin = rangeMin;
  density[0].invRange = invRange;
  overflow[0].rangeMin = rangeMin;
  overflow[0].invRange = invRange;
  surface[0].rangeMin = rangeMin;
  surface[0].invRange = invRange;
}

uint histogramDensityAt(const device atomic_uint* density, int channel, int bin, int width) {
  if (density == nullptr || channel < 0 || bin < 0 || width <= 0) return 0u;
  return atomic_load_explicit(&density[uint(channel * width + bin)], memory_order_relaxed);
}

float smoothedHistogramDensity(const device atomic_uint* density, int channel, int bin, int width) {
  constexpr float weights[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
  float sum = 0.0;
  float weightSum = 0.0;
  for (int tap = -2; tap <= 2; ++tap) {
    int sourceBin = clamp(bin + tap, 0, max(width - 1, 0));
    float weight = weights[tap + 2];
    sum += float(histogramDensityAt(density, channel, sourceBin, width)) * weight;
    weightSum += weight;
  }
  return weightSum > 0.0 ? sum / weightSum : 0.0;
}

kernel void histogramSurfaceMaxKernel(const device atomic_uint* density [[buffer(0)]],
                                      device atomic_uint* maxDensity [[buffer(1)]],
                                      constant HistogramSurfaceUniforms& u [[buffer(2)]],
                                      uint index [[thread_position_in_grid]]) {
  int width = max(u.width, 1);
  int channelCount = max(u.channelCount, 1);
  uint total = uint(width * channelCount);
  if (index >= total || density == nullptr || maxDensity == nullptr) return;
  int channel = int(index / uint(width));
  int bin = int(index % uint(width));
  uint smoothed = uint(round(smoothedHistogramDensity(density, channel, bin, width)));
  atomic_fetch_max_explicit(&maxDensity[0], smoothed, memory_order_relaxed);
}

float4 overColor(float4 dst, float4 src) {
  float invA = 1.0 - clamp(src.a, 0.0, 1.0);
  return float4(src.rgb * src.a + dst.rgb * invA, src.a + dst.a * invA);
}

float4 histogramChannelColor(int channel, bool lumaOnly, bool overflow, bool highlightOverflow) {
  if (overflow) {
    if (highlightOverflow) return float4(0.78, 0.30, 1.0, 0.94);
    if (lumaOnly) return float4(0.74, 0.82, 0.90, 0.72);
    if (channel == 0) return float4(1.0, 0.24, 0.18, 0.76);
    if (channel == 1) return float4(0.26, 1.0, 0.36, 0.76);
    return float4(0.32, 0.58, 1.0, 0.76);
  }
  if (lumaOnly) return float4(0.88, 0.92, 0.96, 0.88);
  if (channel == 0) return float4(1.0, 0.16, 0.12, 0.76);
  if (channel == 1) return float4(0.20, 1.0, 0.28, 0.76);
  return float4(0.24, 0.52, 1.0, 0.76);
}

kernel void histogramSurfaceRenderKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                         const device atomic_uint* density [[buffer(0)]],
                                         const device atomic_uint* overflowDensity [[buffer(1)]],
                                         const device atomic_uint* maxDensity [[buffer(2)]],
                                         constant HistogramSurfaceUniforms& u [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  uint outWidth = outTexture.get_width();
  uint outHeight = outTexture.get_height();
  if (gid.x >= outWidth || gid.y >= outHeight) return;
  float2 uv = float2((float(gid.x) + 0.5) / float(max(outWidth, 1u)),
                     (float(gid.y) + 0.5) / float(max(outHeight, 1u)));
  int width = max(u.width, 1);
  int bin = clamp(int(floor(uv.x * float(width))), 0, width - 1);
  int channelCount = max(u.channelCount, 1);
  bool lumaOnly = u.scopeMode == 1 || channelCount == 1;
  uint maxValue = maxDensity == nullptr ? 1u : atomic_load_explicit(&maxDensity[0], memory_order_relaxed);
  float invMax = 1.0 / max(1.0, float(maxValue));
  float4 color = float4(0.0);
  float lineThickness = max(1.5 / float(max(outHeight, 1u)), 0.003);
  for (int channel = 0; channel < channelCount; ++channel) {
    float densityValue = smoothedHistogramDensity(density, channel, bin, width);
    if (densityValue > 0.0) {
      float curve = sqrt(clamp(densityValue * invMax, 0.0, 1.0)) * 0.965;
      float fillAlpha = uv.y <= curve ? (lumaOnly ? 0.14 : 0.10) : 0.0;
      float lineAlpha = 1.0 - smoothstep(lineThickness, lineThickness * 2.5, abs(uv.y - curve));
      float4 base = histogramChannelColor(channel, lumaOnly, false, false);
      if (fillAlpha > 0.0) color = overColor(color, float4(base.rgb, fillAlpha));
      if (lineAlpha > 0.0) color = overColor(color, float4(base.rgb, base.a * lineAlpha));
    }
    if (u.showOverflow != 0 && overflowDensity != nullptr) {
      float overflowValue = smoothedHistogramDensity(overflowDensity, channel, bin, width);
      if (overflowValue > 0.0) {
        float curve = sqrt(clamp(overflowValue * invMax, 0.0, 1.0)) * 0.965;
        float fillAlpha = uv.y <= curve ? 0.16 : 0.0;
        float lineAlpha = 1.0 - smoothstep(lineThickness, lineThickness * 2.5, abs(uv.y - curve));
        float4 base = histogramChannelColor(channel, lumaOnly, true, u.highlightOverflow != 0);
        if (fillAlpha > 0.0) color = overColor(color, float4(base.rgb, fillAlpha));
        if (lineAlpha > 0.0) color = overColor(color, float4(base.rgb, base.a * lineAlpha));
      }
    }
  }
  outTexture.write(color, gid);
}

uint waveformDensityAt(const device atomic_uint* density, int channel, int x, int y, int width, int height) {
  if (density == nullptr || channel < 0 || x < 0 || y < 0 || width <= 0 || height <= 0) return 0u;
  return atomic_load_explicit(
      &density[uint((channel * width + clamp(x, 0, width - 1)) * height + clamp(y, 0, height - 1))],
      memory_order_relaxed);
}

float smoothedWaveformDensity(const device atomic_uint* density,
                              int channel,
                              int x,
                              int y,
                              int width,
                              int height) {
  constexpr float weights[3] = {1.0, 2.0, 1.0};
  float sum = 0.0;
  float weightSum = 0.0;
  for (int ox = -1; ox <= 1; ++ox) {
    for (int oy = -1; oy <= 1; ++oy) {
      float weight = weights[ox + 1] * weights[oy + 1];
      sum += float(waveformDensityAt(density, channel, x + ox, y + oy, width, height)) * weight;
      weightSum += weight;
    }
  }
  return weightSum > 0.0 ? sum / weightSum : 0.0;
}

kernel void waveformSurfaceMaxKernel(const device atomic_uint* density [[buffer(0)]],
                                     device atomic_uint* maxDensity [[buffer(1)]],
                                     constant WaveformSurfaceUniforms& u [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]) {
  int width = max(u.width, 1);
  int height = max(u.height, 1);
  int channelCount = max(u.channelCount, 1);
  uint total = uint(width * height * channelCount);
  if (index >= total || density == nullptr || maxDensity == nullptr) return;
  int channelStride = width * height;
  int channel = int(index / uint(channelStride));
  int rem = int(index % uint(channelStride));
  int x = rem / height;
  int y = rem % height;
  uint smoothed = uint(round(smoothedWaveformDensity(density, channel, x, y, width, height)));
  atomic_fetch_max_explicit(&maxDensity[0], smoothed, memory_order_relaxed);
}

bool waveformSurfaceChannelVisible(int channel, bool lumaOnly, constant WaveformSurfaceUniforms& u) {
  if (lumaOnly) return channel == 0;
  if (channel == 0) return u.includeRed != 0;
  if (channel == 1) return u.includeGreen != 0;
  if (channel == 2) return u.includeBlue != 0;
  return u.includeLuma != 0;
}

float3 waveformRgbDensityColor(float red, float green, float blue, float saturation) {
  float3 redHue = float3(1.00, 0.12, 0.04);
  float3 greenHue = float3(0.12, 1.00, 0.24);
  float3 blueHue = float3(0.20, 0.46, 1.00);
  float3 color = redHue * red + greenHue * green + blueHue * blue;
  float peak = max(color.r, max(color.g, color.b));
  if (peak > 1.0) color /= peak;
  float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
  float effectiveSaturation = 0.34 + (1.0 - 0.34) * clamp(saturation, 0.0, 1.0);
  return clamp(mix(float3(luma), color, effectiveSaturation), 0.0, 1.0);
}

float4 waveformChannelColor(int channel, bool lumaOnly, float intensity, constant WaveformSurfaceUniforms& u) {
  if (lumaOnly || channel == 3) return float4(0.88 * intensity, 0.92 * intensity, 0.96 * intensity, 1.0);
  return float4(waveformRgbDensityColor(channel == 0 ? intensity : 0.0,
                                        channel == 1 ? intensity : 0.0,
                                        channel == 2 ? intensity : 0.0,
                                        u.colorSaturation),
                1.0);
}

kernel void waveformSurfaceRenderKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                        const device atomic_uint* density [[buffer(0)]],
                                        const device atomic_uint* overflowDensity [[buffer(1)]],
                                        const device atomic_uint* maxDensity [[buffer(2)]],
                                        constant WaveformSurfaceUniforms& u [[buffer(3)]],
                                        uint2 gid [[thread_position_in_grid]]) {
  uint outWidth = outTexture.get_width();
  uint outHeight = outTexture.get_height();
  if (gid.x >= outWidth || gid.y >= outHeight) return;
  float2 uv = float2((float(gid.x) + 0.5) / float(max(outWidth, 1u)),
                     (float(gid.y) + 0.5) / float(max(outHeight, 1u)));
  int width = max(u.width, 1);
  int height = max(u.height, 1);
  int channelCount = max(u.channelCount, 1);
  bool lumaOnly = u.scopeMode == 2 || channelCount == 1;
  uint maxValue = maxDensity == nullptr ? 1u : atomic_load_explicit(&maxDensity[0], memory_order_relaxed);
  float invLogMax = 1.0 / log(1.0 + max(1.0, float(maxValue)));
  float4 color = float4(0.0);

  if (u.scopeMode == 1 && !lumaOnly) {
    float scaledX = uv.x * float(channelCount);
    int channel = clamp(int(floor(scaledX)), 0, channelCount - 1);
    if (waveformSurfaceChannelVisible(channel, false, u)) {
      float channelX = fract(scaledX);
      int x = clamp(int(floor(channelX * float(width))), 0, width - 1);
      int y = clamp(height - 1 - int(floor(uv.y * float(height))), 0, height - 1);
      float value = smoothedWaveformDensity(density, channel, x, y, width, height);
      if (value > 0.0) {
        float normalized = clamp(log(1.0 + value) * invLogMax, 0.0, 1.0);
        float intensity = clamp((0.20 + 0.80 * pow(normalized, 0.62)) * u.pointBrightness, 0.0, 1.0);
        float4 base = waveformChannelColor(channel, false, intensity, u);
        color = overColor(color, float4(base.rgb, clamp((0.18 + 0.82 * intensity) * u.coverageAlpha, 0.0, 1.0)));
      }
      if (u.showOverflow != 0 && overflowDensity != nullptr) {
        float overflowValue = smoothedWaveformDensity(overflowDensity, channel, x, y, width, height);
        if (overflowValue > 0.0) {
          float normalized = clamp(log(1.0 + overflowValue) * invLogMax, 0.0, 1.0);
          float intensity = clamp((0.28 + 0.72 * normalized) * u.pointBrightness, 0.0, 1.0);
          float4 base = histogramChannelColor(channel, channel == 3, true, u.highlightOverflow != 0);
          color = overColor(color, float4(base.rgb * intensity, clamp(0.82 * u.coverageAlpha, 0.0, 1.0)));
        }
      }
    }
  } else {
    int x = clamp(int(floor(uv.x * float(width))), 0, width - 1);
    int y = clamp(height - 1 - int(floor(uv.y * float(height))), 0, height - 1);
    if (lumaOnly) {
      float value = smoothedWaveformDensity(density, 0, x, y, width, height);
      if (value > 0.0) {
        float normalized = clamp(log(1.0 + value) * invLogMax, 0.0, 1.0);
        float intensity = clamp((0.20 + 0.80 * pow(normalized, 0.62)) * u.pointBrightness, 0.0, 1.0);
        color = overColor(color,
                          float4(0.88 * intensity, 0.92 * intensity, 0.96 * intensity,
                                 clamp((0.18 + 0.82 * intensity) * u.coverageAlpha, 0.0, 1.0)));
      }
    } else {
      float red = u.includeRed != 0 ? smoothedWaveformDensity(density, 0, x, y, width, height) : 0.0;
      float green = u.includeGreen != 0 ? smoothedWaveformDensity(density, 1, x, y, width, height) : 0.0;
      float blue = u.includeBlue != 0 ? smoothedWaveformDensity(density, 2, x, y, width, height) : 0.0;
      float3 intensities = float3(red, green, blue);
      if (intensities.r > 0.0 || intensities.g > 0.0 || intensities.b > 0.0) {
        intensities.r = clamp((0.20 + 0.80 * pow(clamp(log(1.0 + intensities.r) * invLogMax, 0.0, 1.0), 0.62)) *
                                  u.pointBrightness,
                              0.0,
                              1.0);
        intensities.g = clamp((0.20 + 0.80 * pow(clamp(log(1.0 + intensities.g) * invLogMax, 0.0, 1.0), 0.62)) *
                                  u.pointBrightness,
                              0.0,
                              1.0);
        intensities.b = clamp((0.20 + 0.80 * pow(clamp(log(1.0 + intensities.b) * invLogMax, 0.0, 1.0), 0.62)) *
                                  u.pointBrightness,
                              0.0,
                              1.0);
        float3 rgb = waveformRgbDensityColor(intensities.r, intensities.g, intensities.b, u.colorSaturation);
        float alpha = clamp((0.16 + 0.84 * max(max(intensities.r, intensities.g), intensities.b)) *
                                u.coverageAlpha,
                            0.0,
                            1.0);
        color = overColor(color, float4(rgb, alpha));
      }
    }
    if (u.showOverflow != 0 && overflowDensity != nullptr) {
      for (int channel = 0; channel < channelCount; ++channel) {
        if (!waveformSurfaceChannelVisible(channel, lumaOnly, u)) continue;
        float overflowValue = smoothedWaveformDensity(overflowDensity, channel, x, y, width, height);
        if (overflowValue <= 0.0) continue;
        float normalized = clamp(log(1.0 + overflowValue) * invLogMax, 0.0, 1.0);
        float intensity = clamp((0.28 + 0.72 * normalized) * u.pointBrightness, 0.0, 1.0);
        float4 base = histogramChannelColor(channel, lumaOnly || channel == 3, true, u.highlightOverflow != 0);
        color = overColor(color, float4(base.rgb * intensity, clamp(0.82 * u.coverageAlpha, 0.0, 1.0)));
      }
    }
  }
  outTexture.write(color, gid);
}

int glossNeighborhoodRadiusCells(int neighborhoodChoice) {
  switch (clamp(neighborhoodChoice, 0, 2)) {
    case 0: return 1;
    case 2: return 3;
    case 1:
    default: return 2;
  }
}

int glossAnalysisRadiusCells(int neighborhoodChoice) {
  return max(2, glossNeighborhoodRadiusCells(neighborhoodChoice) * 2);
}

float sampleGridClamped(const device float* values, int width, int height, int x, int y) {
  if (values == nullptr || width <= 0 || height <= 0) return 0.0;
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  return values[uint(y * width + x)];
}

float glossGradientMagnitude(const device float* values, int width, int height, int x, int y) {
  float gx = 0.5 * (sampleGridClamped(values, width, height, x + 1, y) -
                    sampleGridClamped(values, width, height, x - 1, y));
  float gy = 0.5 * (sampleGridClamped(values, width, height, x, y + 1) -
                    sampleGridClamped(values, width, height, x, y - 1));
  return sqrt(gx * gx + gy * gy);
}

kernel void glossFieldAccumulateKernel(const device float* packedPoints [[buffer(0)]],
                                       device atomic_uint* occupancyCounts [[buffer(1)]],
                                       device atomic_uint* sumR [[buffer(2)]],
                                       device atomic_uint* sumG [[buffer(3)]],
                                       device atomic_uint* sumB [[buffer(4)]],
                                       device atomic_uint* sumY [[buffer(5)]],
                                       device atomic_uint* sumMax [[buffer(6)]],
                                       device atomic_uint* sumMin [[buffer(7)]],
                                       device atomic_uint* sumNeutrality [[buffer(8)]],
                                       constant GlossFieldAccumulateUniforms& u [[buffer(9)]],
                                       uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total) return;
  uint base = index * 6u;
  float xNorm = clamp(packedPoints[base + 0u], 0.0, 1.0);
  float yNorm = clamp(packedPoints[base + 1u], 0.0, 1.0);
  float r = packedPoints[base + 3u];
  float g = packedPoints[base + 4u];
  float b = packedPoints[base + 5u];
  if (u.showOverflow == 0) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  float maxRgb = clamp(max(r, max(g, b)), 0.0, 1.0);
  float minRgb = clamp(max(0.0, min(r, min(g, b))), 0.0, 1.0);
  float neutralityValue = maxRgb > 1e-6 ? clamp(minRgb / maxRgb, 0.0, 1.0) : 0.0;
  float luma = clamp(r * 0.2126 + g * 0.7152 + b * 0.0722, 0.0, 1.0);
  int x = clamp(int(xNorm * float(u.gridWidth)), 0, max(u.gridWidth - 1, 0));
  int y = clamp(int((1.0 - yNorm) * float(u.gridHeight)), 0, max(u.gridHeight - 1, 0));
  uint cellIndex = uint(y * u.gridWidth + x);
  atomic_fetch_add_explicit(&occupancyCounts[cellIndex], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&sumR[cellIndex], glossEncodeAccum(r), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumG[cellIndex], glossEncodeAccum(g), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumB[cellIndex], glossEncodeAccum(b), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumY[cellIndex], glossEncodeAccum(luma), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMax[cellIndex], glossEncodeAccum(maxRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMin[cellIndex], glossEncodeAccum(minRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumNeutrality[cellIndex], glossEncodeAccum(neutralityValue), memory_order_relaxed);
}

kernel void glossFieldFinalizeKernel(const device atomic_uint* occupancyCounts [[buffer(0)]],
                                     const device atomic_uint* sumR [[buffer(1)]],
                                     const device atomic_uint* sumG [[buffer(2)]],
                                     const device atomic_uint* sumB [[buffer(3)]],
                                     const device atomic_uint* sumY [[buffer(4)]],
                                     const device atomic_uint* sumMax [[buffer(5)]],
                                     const device atomic_uint* sumMin [[buffer(6)]],
                                     const device atomic_uint* sumNeutrality [[buffer(7)]],
                                     device float* occupancy [[buffer(8)]],
                                     device float* meanR [[buffer(9)]],
                                     device float* meanG [[buffer(10)]],
                                     device float* meanB [[buffer(11)]],
                                     device float* carrierY [[buffer(12)]],
                                     device float* carrierMax [[buffer(13)]],
                                     device float* carrierMin [[buffer(14)]],
                                     device float* neutrality [[buffer(15)]],
                                     constant GlossFieldCellUniforms& u [[buffer(16)]],
                                     uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  uint count = atomic_load_explicit(&occupancyCounts[index], memory_order_relaxed);
  occupancy[index] = float(count);
  if (count == 0u) {
    meanR[index] = 0.0;
    meanG[index] = 0.0;
    meanB[index] = 0.0;
    carrierY[index] = 0.0;
    carrierMax[index] = 0.0;
    carrierMin[index] = 0.0;
    neutrality[index] = 0.0;
    return;
  }
  float invCount = 1.0 / float(count);
  meanR[index] = glossDecodeAccum(atomic_load_explicit(&sumR[index], memory_order_relaxed)) * invCount;
  meanG[index] = glossDecodeAccum(atomic_load_explicit(&sumG[index], memory_order_relaxed)) * invCount;
  meanB[index] = glossDecodeAccum(atomic_load_explicit(&sumB[index], memory_order_relaxed)) * invCount;
  carrierY[index] = glossDecodeAccum(atomic_load_explicit(&sumY[index], memory_order_relaxed)) * invCount;
  carrierMax[index] = glossDecodeAccum(atomic_load_explicit(&sumMax[index], memory_order_relaxed)) * invCount;
  carrierMin[index] = glossDecodeAccum(atomic_load_explicit(&sumMin[index], memory_order_relaxed)) * invCount;
  neutrality[index] = glossDecodeAccum(atomic_load_explicit(&sumNeutrality[index], memory_order_relaxed)) * invCount;
}

kernel void glossFieldMaxKernel(const device float* values [[buffer(0)]],
                                device atomic_uint* outBits [[buffer(1)]],
                                constant GlossFieldCellUniforms& u [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  atomic_fetch_max_explicit(&outBits[0], as_type<uint>(max(values[index], 0.0)), memory_order_relaxed);
}

kernel void glossFieldNormalizeKernel(const device float* src [[buffer(0)]],
                                      device float* dst [[buffer(1)]],
                                      const device atomic_uint* maxBits [[buffer(2)]],
                                      constant GlossFieldCellUniforms& u [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  float denom = max(as_type<float>(atomic_load_explicit(&maxBits[0], memory_order_relaxed)), 1e-5);
  dst[index] = clamp(src[index] / denom, 0.0, 1.0);
}

kernel void glossFieldBlurKernel(const device float* src [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 constant GlossFieldCellUniforms& u [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float accum = 0.0;
  float weight = 0.0;
  for (int oy = -1; oy <= 1; ++oy) {
    int yy = y + oy;
    if (yy < 0 || yy >= u.gridHeight) continue;
    for (int ox = -1; ox <= 1; ++ox) {
      int xx = x + ox;
      if (xx < 0 || xx >= u.gridWidth) continue;
      float tapWeight = (ox == 0 && oy == 0) ? 0.30 : ((ox == 0 || oy == 0) ? 0.13 : 0.08);
      accum += src[uint(yy * u.gridWidth + xx)] * tapWeight;
      weight += tapWeight;
    }
  }
  dst[index] = weight > 1e-6 ? (accum / weight) : 0.0;
}

kernel void glossFieldBodyKernel(const device float* occupancy [[buffer(0)]],
                                 const device float* meanR [[buffer(1)]],
                                 const device float* meanG [[buffer(2)]],
                                 const device float* meanB [[buffer(3)]],
                                 const device float* carrierMax [[buffer(4)]],
                                 device float* body [[buffer(5)]],
                                 constant GlossFieldCellUniforms& u [[buffer(6)]],
                                 uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    body[index] = 0.0;
    return;
  }
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  int radiusCells = glossNeighborhoodRadiusCells(u.neighborhoodChoice);
  const int kMaxNeighborhood = 49;
  float carriers[kMaxNeighborhood];
  int neighborIndices[kMaxNeighborhood];
  int count = 0;
  float centerCarrier = carrierMax[index];
  float centerR = meanR[index];
  float centerG = meanG[index];
  float centerB = meanB[index];
  for (int oy = -radiusCells; oy <= radiusCells; ++oy) {
    int yy = y + oy;
    if (yy < 0 || yy >= u.gridHeight) continue;
    for (int ox = -radiusCells; ox <= radiusCells; ++ox) {
      int xx = x + ox;
      if (xx < 0 || xx >= u.gridWidth) continue;
      uint neighborIndex = uint(yy * u.gridWidth + xx);
      if (occupancy[neighborIndex] <= 0.5) continue;
      float carrier = carrierMax[neighborIndex];
      float dr = meanR[neighborIndex] - centerR;
      float dg = meanG[neighborIndex] - centerG;
      float db = meanB[neighborIndex] - centerB;
      float colorDistance = sqrt(dr * dr + dg * dg + db * db);
      if (abs(carrier - centerCarrier) > 0.26 && colorDistance > 0.20) continue;
      if (count < kMaxNeighborhood) {
        carriers[count] = carrier;
        neighborIndices[count] = int(neighborIndex);
        ++count;
      }
    }
  }
  if (count <= 0) {
    body[index] = centerCarrier;
    return;
  }
  for (int i = 1; i < count; ++i) {
    float keyCarrier = carriers[i];
    int keyIndex = neighborIndices[i];
    int j = i - 1;
    while (j >= 0 && (carriers[j] > keyCarrier || (carriers[j] == keyCarrier && neighborIndices[j] > keyIndex))) {
      carriers[j + 1] = carriers[j];
      neighborIndices[j + 1] = neighborIndices[j];
      --j;
    }
    carriers[j + 1] = keyCarrier;
    neighborIndices[j + 1] = keyIndex;
  }
  int trim = count >= 6 ? max(1, count / 6) : 0;
  int begin = min(trim, count);
  int end = max(begin + 1, count - trim);
  float bodySum = 0.0;
  float bodyWeight = 0.0;
  for (int i = begin; i < end; ++i) {
    int neighborIndex = neighborIndices[i];
    int neighborX = neighborIndex % u.gridWidth;
    int neighborY = neighborIndex / u.gridWidth;
    float dx = float(neighborX - x);
    float dy = float(neighborY - y);
    float spatialWeight = 1.0 / (1.0 + dx * dx + dy * dy);
    bodySum += carriers[i] * spatialWeight;
    bodyWeight += spatialWeight;
  }
  body[index] = bodyWeight > 1e-6 ? (bodySum / bodyWeight) : centerCarrier;
}

kernel void glossFieldRawSignalKernel(const device float* occupancy [[buffer(0)]],
                                      const device float* carrierMax [[buffer(1)]],
                                      const device float* body [[buffer(2)]],
                                      device float* rawSignal [[buffer(3)]],
                                      device atomic_uint* maxBits [[buffer(4)]],
                                      constant GlossFieldCellUniforms& u [[buffer(5)]],
                                      uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    rawSignal[index] = 0.0;
    return;
  }
  float bodyValue = max(body[index], 0.0);
  float rawPositive = max(0.0, carrierMax[index] - bodyValue);
  float rawNegative = max(0.0, bodyValue - carrierMax[index]);
  rawSignal[index] = rawPositive - rawNegative;
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(bodyValue), memory_order_relaxed);
}

kernel void glossFieldWeightedSignalKernel(const device float* occupancyNorm [[buffer(0)]],
                                           const device float* body [[buffer(1)]],
                                           const device float* rawSignal [[buffer(2)]],
                                           device float* positive [[buffer(3)]],
                                           device float* negative [[buffer(4)]],
                                           device float* boundary [[buffer(5)]],
                                           device float* congruence [[buffer(6)]],
                                           device float* confidence [[buffer(7)]],
                                           device float* signal [[buffer(8)]],
                                           device atomic_uint* maxBits [[buffer(9)]],
                                           constant GlossFieldCellUniforms& u [[buffer(10)]],
                                           uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float occCenter = sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y);
  if (occCenter <= 0.0) {
    positive[index] = 0.0;
    negative[index] = 0.0;
    boundary[index] = 0.0;
    congruence[index] = 0.0;
    confidence[index] = 0.0;
    signal[index] = 0.0;
    return;
  }
  float gxCarrier = sampleGridClamped(body, u.gridWidth, u.gridHeight, x + 1, y) -
                    sampleGridClamped(body, u.gridWidth, u.gridHeight, x - 1, y);
  float gyCarrier = sampleGridClamped(body, u.gridWidth, u.gridHeight, x, y + 1) -
                    sampleGridClamped(body, u.gridWidth, u.gridHeight, x, y - 1);
  float gxSignal = sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x + 1, y) -
                   sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x - 1, y);
  float gySignal = sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x, y + 1) -
                   sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x, y - 1);
  float magCarrier = sqrt(gxCarrier * gxCarrier + gyCarrier * gyCarrier);
  float magSignal = sqrt(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0;
  if (magCarrier > 1e-6 && magSignal > 1e-6) {
    localCongruence = abs((gxCarrier * gxSignal + gyCarrier * gySignal) / (magCarrier * magSignal));
  } else if (magSignal > 1e-6) {
    localCongruence = 0.35;
  }
  float occNeighborhood =
      (occCenter +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x + 1, y) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x - 1, y) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y + 1) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y - 1)) / 5.0;
  float localConfidence = clamp(sqrt(occCenter) * clamp(0.28 + 0.72 * occNeighborhood, 0.0, 1.0), 0.0, 1.0);
  float posWeighted = max(0.0, rawSignal[index]) * (0.30 + 0.70 * localCongruence) * localConfidence;
  float negWeighted = max(0.0, -rawSignal[index]) * (0.30 + 0.70 * localCongruence) * localConfidence;
  float boundaryValue = clamp(magSignal * 4.0, 0.0, 1.0) * localConfidence;
  positive[index] = posWeighted;
  negative[index] = negWeighted;
  boundary[index] = boundaryValue;
  congruence[index] = localCongruence;
  confidence[index] = localConfidence;
  signal[index] = posWeighted - negWeighted;
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(max(posWeighted, 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[1], as_type<uint>(max(negWeighted, 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[2], as_type<uint>(max(boundaryValue, 0.0)), memory_order_relaxed);
}

kernel void glossFieldMergeMaxBitsKernel(const device atomic_uint* bodyMaxBits [[buffer(0)]],
                                         const device atomic_uint* weightedMaxBits [[buffer(1)]],
                                         device atomic_uint* finalMaxBits [[buffer(2)]],
                                         uint index [[thread_position_in_grid]]) {
  if (index != 0u) return;
  atomic_store_explicit(&finalMaxBits[0],
                        atomic_load_explicit(&bodyMaxBits[0], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[1],
                        atomic_load_explicit(&weightedMaxBits[0], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[2],
                        atomic_load_explicit(&weightedMaxBits[1], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[3],
                        atomic_load_explicit(&weightedMaxBits[2], memory_order_relaxed),
                        memory_order_relaxed);
}

kernel void glossFieldFinalNormalizeKernel(device float* body [[buffer(0)]],
                                           device float* signal [[buffer(1)]],
                                           device float* positive [[buffer(2)]],
                                           device float* negative [[buffer(3)]],
                                           device float* boundary [[buffer(4)]],
                                           const device atomic_uint* maxBits [[buffer(5)]],
                                           constant GlossFieldCellUniforms& u [[buffer(6)]],
                                           uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  float maxBody = max(as_type<float>(atomic_load_explicit(&maxBits[0], memory_order_relaxed)), 1e-5);
  float maxPositive = max(as_type<float>(atomic_load_explicit(&maxBits[1], memory_order_relaxed)), 1e-5);
  float maxNegative = max(as_type<float>(atomic_load_explicit(&maxBits[2], memory_order_relaxed)), 1e-5);
  float maxBoundary = max(as_type<float>(atomic_load_explicit(&maxBits[3], memory_order_relaxed)), 1e-5);
  float maxAbsSignal = max(max(maxPositive, maxNegative), 1e-5);
  body[index] = clamp(body[index] / maxBody, 0.0, 1.0);
  positive[index] = clamp(positive[index] / maxPositive, 0.0, 1.0);
  negative[index] = clamp(negative[index] / maxNegative, 0.0, 1.0);
  signal[index] = clamp(signal[index] / maxAbsSignal, -1.0, 1.0);
  boundary[index] = clamp(boundary[index] / maxBoundary, 0.0, 1.0);
}

kernel void glossFieldLocalPercentileKernel(const device float* values [[buffer(0)]],
                                            const device float* occupancy [[buffer(1)]],
                                            device float* outValues [[buffer(2)]],
                                            constant GlossFieldCellUniforms& u [[buffer(3)]],
                                            constant float& percentile [[buffer(4)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  if (occupancy != nullptr && occupancy[index] <= 0.5) {
    outValues[index] = 0.0;
    return;
  }
  int radius = max(1, glossAnalysisRadiusCells(u.neighborhoodChoice));
  int bins[96];
  for (int i = 0; i < 96; ++i) bins[i] = 0;
  int count = 0;
  float minValue = 1.0e20;
  float maxValue = -1.0e20;
  for (int yy = max(0, y - radius); yy <= min(u.gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(u.gridWidth - 1, x + radius); ++xx) {
      uint nidx = uint(yy * u.gridWidth + xx);
      if (occupancy != nullptr && occupancy[nidx] <= 0.5) continue;
      float v = values[nidx];
      minValue = min(minValue, v);
      maxValue = max(maxValue, v);
      ++count;
    }
  }
  if (count <= 0 || maxValue - minValue <= 1e-7) {
    outValues[index] = values[index];
    return;
  }
  float invRange = 1.0 / (maxValue - minValue);
  for (int yy = max(0, y - radius); yy <= min(u.gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(u.gridWidth - 1, x + radius); ++xx) {
      uint nidx = uint(yy * u.gridWidth + xx);
      if (occupancy != nullptr && occupancy[nidx] <= 0.5) continue;
      int bin = min(95,
                    max(0, int((values[nidx] - minValue) * invRange *
                               95.0 + 0.5)));
      bins[bin] += 1;
    }
  }
  int target = min(count - 1,
                   max(0, int(floor(clamp(percentile / 100.0, 0.0, 1.0) *
                                    float(count - 1) + 0.5))));
  int accumulated = 0;
  int chosen = 0;
  for (int i = 0; i < 96; ++i) {
    accumulated += bins[i];
    if (accumulated > target) {
      chosen = i;
      break;
    }
  }
  outValues[index] = minValue + (float(chosen) / 95.0) * (maxValue - minValue);
}

kernel void glossFieldCandidate2RawKernel(const device float* occupancy [[buffer(0)]],
                                          const device float* occupancySupport [[buffer(1)]],
                                          const device float* meanR [[buffer(2)]],
                                          const device float* meanG [[buffer(3)]],
                                          const device float* meanB [[buffer(4)]],
                                          const device float* carrier [[buffer(5)]],
                                          const device float* viewerBody [[buffer(6)]],
                                          const device float* bodyCore [[buffer(7)]],
                                          const device float* bodyContext [[buffer(8)]],
                                          const device float* retinexBody [[buffer(9)]],
                                          const device float* dogLow [[buffer(10)]],
                                          const device float* dogHigh [[buffer(11)]],
                                          device float* adaptiveBody [[buffer(12)]],
                                          device float* positiveRaw [[buffer(13)]],
                                          device float* negativeRaw [[buffer(14)]],
                                          device float* confidence [[buffer(15)]],
                                          device float* agreement [[buffer(16)]],
                                          constant GlossFieldCellUniforms& u [[buffer(17)]],
                                          uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    adaptiveBody[index] = 0.0;
    positiveRaw[index] = 0.0;
    negativeRaw[index] = 0.0;
    confidence[index] = 0.0;
    agreement[index] = 0.0;
    return;
  }
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  int analysisRadius = glossAnalysisRadiusCells(u.neighborhoodChoice);
  float hybridBody = 0.65 * bodyCore[index] + 0.35 * bodyContext[index];
  float viewerPositive = max(0.0, carrier[index] - viewerBody[index]);
  float viewerNegative = max(0.0, viewerBody[index] - carrier[index]);
  float hybridPositive = max(0.0, carrier[index] - bodyCore[index]);
  float hybridNegative = max(0.0, bodyContext[index] - carrier[index]);
  float chromaSpread = max(meanR[index], max(meanG[index], meanB[index])) -
                       min(meanR[index], min(meanG[index], meanB[index]));
  float shapeSupport = clamp(glossGradientMagnitude(bodyContext, u.gridWidth, u.gridHeight, x, y) * 8.0, 0.0, 1.0);
  float chromaSupport = clamp(chromaSpread * 2.5, 0.0, 1.0);
  float ambiguity = clamp(1.0 - (0.72 * shapeSupport + 0.28 * chromaSupport), 0.0, 1.0);
  float body = ambiguity * viewerBody[index] + (1.0 - ambiguity) * hybridBody;
  float bodyAgreement = clamp(1.0 - abs(viewerBody[index] - hybridBody), 0.0, 1.0);
  float positiveAgreement = clamp(1.0 - abs(viewerPositive - hybridPositive) * 4.0, 0.0, 1.0);
  float localPositiveSupport = 0.0;
  float supportWeight = 0.0;
  for (int oy = -analysisRadius; oy <= analysisRadius; ++oy) {
    for (int ox = -analysisRadius; ox <= analysisRadius; ++ox) {
      int xx = clamp(x + ox, 0, u.gridWidth - 1);
      int yy = clamp(y + oy, 0, u.gridHeight - 1);
      uint nidx = uint(yy * u.gridWidth + xx);
      float distSq = float(ox * ox + oy * oy);
      float w = 1.0 / (1.0 + distSq);
      localPositiveSupport += max(0.0, carrier[nidx] - bodyCore[nidx]) * w;
      supportWeight += w;
    }
  }
  localPositiveSupport = clamp((supportWeight > 1e-6 ? localPositiveSupport / supportWeight : 0.0) * 4.0, 0.0, 1.0);
  float permission = clamp(0.32 * positiveAgreement +
                           0.24 * bodyAgreement +
                           0.24 * shapeSupport +
                           0.20 * localPositiveSupport,
                           0.0,
                           1.0);
  float consensusPositive = viewerPositive * (0.25 + 0.75 * clamp(hybridPositive * 4.0, 0.0, 1.0));
  float panelMix = clamp(sqrt(max(0.0, ambiguity)) * (0.55 + 0.45 * positiveAgreement), 0.0, 1.0);
  float retinexResidual = carrier[index] - retinexBody[index];
  float dogResidual = dogLow[index] - dogHigh[index];
  float dogPositive = max(0.0, dogResidual);
  float dogNegative = max(0.0, -dogResidual);
  float dogPositiveAgreement = clamp(1.0 - abs(dogPositive - hybridPositive) * 4.0, 0.0, 1.0);
  float dogNegativeAgreement = clamp(1.0 - abs(dogNegative - hybridNegative) * 4.0, 0.0, 1.0);
  float positiveRetinexGate = clamp(0.18 + 0.34 * permission + 0.18 * positiveAgreement +
                                    0.16 * shapeSupport + 0.14 * localPositiveSupport,
                                    0.0,
                                    1.0);
  float negativeRetinexGate = clamp(0.30 + 0.40 * (1.0 - ambiguity) +
                                    0.18 * bodyAgreement + 0.12 * permission,
                                    0.0,
                                    1.0);
  float dogPositiveGate = clamp(0.16 + 0.30 * permission + 0.18 * positiveAgreement +
                                0.16 * shapeSupport + 0.12 * dogPositiveAgreement +
                                0.08 * localPositiveSupport,
                                0.0,
                                1.0);
  float dogNegativeGate = clamp(0.30 + 0.36 * (1.0 - ambiguity) +
                                0.20 * bodyAgreement + 0.14 * dogNegativeAgreement,
                                0.0,
                                1.0);
  float pos = (1.0 - panelMix) * hybridPositive +
              panelMix * consensusPositive +
              0.20 * positiveRetinexGate * max(0.0, retinexResidual) +
              0.18 * dogPositiveGate * dogPositive;
  float neg = (1.0 - ambiguity) * hybridNegative +
              ambiguity * (0.55 * hybridNegative + 0.45 * viewerNegative) +
              0.16 * negativeRetinexGate * max(0.0, -retinexResidual) +
              0.12 * dogNegativeGate * dogNegative;
  adaptiveBody[index] = body;
  positiveRaw[index] = pos;
  negativeRaw[index] = neg;
  float attachment = clamp(0.31 * shapeSupport +
                           0.21 * localPositiveSupport +
                           0.20 * permission +
                           0.20 * positiveAgreement +
                           0.08 * bodyAgreement,
                           0.0,
                           1.0);
  float support = sqrt(clamp(occupancySupport[index], 0.0, 1.0));
  confidence[index] = clamp((0.10 + 0.90 * (0.28 * bodyAgreement +
                                            0.22 * positiveAgreement +
                                            0.20 * permission +
                                            0.15 * (1.0 - ambiguity) +
                                            0.15 * attachment)) *
                            (0.30 + 0.70 * support),
                            0.0,
                            1.0);
  agreement[index] = clamp(0.40 * bodyAgreement + 0.35 * positiveAgreement + 0.25 * permission, 0.0, 1.0);
}

kernel void glossFieldAssembleUnifiedKernel(const device float* bodyRaw [[buffer(0)]],
                                            const device float* positiveRaw [[buffer(1)]],
                                            const device float* negativeRaw [[buffer(2)]],
                                            const device float* confidenceRaw [[buffer(3)]],
                                            const device float* agreementRaw [[buffer(4)]],
                                            device float* body [[buffer(5)]],
                                            device float* signal [[buffer(6)]],
                                            device float* positive [[buffer(7)]],
                                            device float* negative [[buffer(8)]],
                                            device float* boundary [[buffer(9)]],
                                            device float* congruence [[buffer(10)]],
                                            device float* confidence [[buffer(11)]],
                                            device atomic_uint* maxBits [[buffer(12)]],
                                            constant GlossFieldCellUniforms& u [[buffer(13)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float gxBody = 0.5 * (sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x + 1, y) -
                        sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x - 1, y));
  float gyBody = 0.5 * (sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x, y + 1) -
                        sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x, y - 1));
  float signalXp = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x + 1, y) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x + 1, y);
  float signalXm = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x - 1, y) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x - 1, y);
  float signalYp = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x, y + 1) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x, y + 1);
  float signalYm = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x, y - 1) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x, y - 1);
  float gxSignal = 0.5 * (signalXp - signalXm);
  float gySignal = 0.5 * (signalYp - signalYm);
  float magBody = sqrt(gxBody * gxBody + gyBody * gyBody);
  float magSignal = sqrt(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0;
  if (magBody > 1e-6 && magSignal > 1e-6) {
    localCongruence = clamp(abs((gxBody * gxSignal + gyBody * gySignal) / (magBody * magSignal)), 0.0, 1.0);
  } else if (magSignal > 1e-6) {
    localCongruence = 0.35;
  }
  float localBoundary = magSignal * 4.0;
  float weight = (0.35 + 0.65 * clamp(localCongruence, 0.0, 1.0)) *
                 (0.45 + 0.55 * clamp(confidenceRaw[index], 0.0, 1.0));
  body[index] = max(0.0, bodyRaw[index]);
  positive[index] = max(0.0, positiveRaw[index]) * weight;
  negative[index] = max(0.0, negativeRaw[index]) * weight;
  signal[index] = positive[index] - negative[index];
  boundary[index] = max(0.0, localBoundary);
  congruence[index] = localCongruence;
  confidence[index] = confidenceRaw[index];
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(max(body[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[1], as_type<uint>(max(positive[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[2], as_type<uint>(max(negative[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[3], as_type<uint>(max(boundary[index], 0.0)), memory_order_relaxed);
}

struct PlotSurfaceClearUniforms {
  float r;
  float g;
  float b;
  float a;
};

struct SourceSignalSurfaceUniforms {
  int sourceWidth;
  int sourceHeight;
  int outputWidth;
  int outputHeight;
  float backgroundR;
  float backgroundG;
  float backgroundB;
  float pad0;
};

struct RasterPointSurfaceUniforms {
  float modelView[16];
  float projection[16];
  float pointRadiusPixels;
  float surfaceWidth;
  float surfaceHeight;
  float pad0;
};

struct SurfaceCompositeUniforms {
  float dstX;
  float dstY;
  float dstW;
  float dstH;
  float drawableW;
  float drawableH;
  float opacity;
  float pad0;
};

struct FrameSolidRectUniforms {
  float dstX;
  float dstY;
  float dstW;
  float dstH;
  float drawableW;
  float drawableH;
  float r;
  float g;
  float b;
  float a;
  float pad0;
  float pad1;
};

struct FrameTextUniforms {
  float drawableW;
  float drawableH;
  float r;
  float g;
  float b;
  float a;
  float clipX;
  float clipY;
  float clipW;
  float clipH;
  float clipEnabled;
  float pad0;
  float pad1;
  float pad2;
};

struct FrameTextVertexIn {
  float x;
  float y;
  float u;
  float v;
};

struct FrameUiVectorUniforms {
  float drawableW;
  float drawableH;
  float pad0;
  float pad1;
};

struct FrameUiVectorVertexIn {
  float x;
  float y;
  float r;
  float g;
  float b;
  float a;
};

struct SurfaceCompositeVertexOut {
  float4 position [[position]];
  float2 uv;
};

struct FrameUiVectorVertexOut {
  float4 position [[position]];
  float4 color;
};

struct RasterPointSurfaceVertexOut {
  float4 position [[position]];
  float4 color;
  float pointSize [[point_size]];
};

struct FrameTextVertexOut {
  float4 position [[position]];
  float2 uv;
  float2 pixel;
};

struct GlossFieldSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
};

struct GlossProjectionSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float sourceAspect;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
  float pointRadiusPixels;
  float modelView[16];
  float projection[16];
};

float3 mixGloss3(float3 a, float3 b, float t) {
  float k = clamp(t, 0.0, 1.0);
  return a + (b - a) * k;
}

float glossSourcePresence(const device float* meanR,
                          const device float* meanG,
                          const device float* meanB,
                          uint idx) {
  return clamp(max(meanR[idx], max(meanG[idx], meanB[idx])), 0.0, 1.0);
}

float3 glossSurfaceSourceHueColor(const device float* meanR,
                                  const device float* meanG,
                                  const device float* meanB,
                                  constant GlossFieldSurfaceUniforms& u,
                                  uint idx) {
  float sr = 0.0;
  float sg = 0.0;
  float sb = 0.0;
  mapDisplayColor(meanR[idx], meanG[idx], meanB[idx], sr, sg, sb);
  applyDisplaySaturation(min(3.0, u.colorSaturation), sr, sg, sb);
  return float3(sr, sg, sb);
}

float4 glossSurfaceUnderlayColor(const device float* meanR,
                                 const device float* meanG,
                                 const device float* meanB,
                                 const device float* body,
                                 const device float* confidence,
                                 constant GlossFieldSurfaceUniforms& u,
                                 uint idx) {
  float sr = 0.0;
  float sg = 0.0;
  float sb = 0.0;
  mapDisplayColor(meanR[idx], meanG[idx], meanB[idx], sr, sg, sb);
  float sourceLuma = clamp(0.2126 * sr + 0.7152 * sg + 0.0722 * sb, 0.0, 1.0);
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, idx);
  float confidenceValue = clamp(confidence[idx], 0.0, 1.0);
  float bodyValue = clamp(body[idx], 0.0, 1.0);
  float structure = max(sqrt(confidenceValue), sqrt(sourcePresence));
  float bodyGain = 0.34 + 0.66 * clamp(u.glossBodyOpacity, 0.0, 1.0);
  float3 color;
  if (u.colorMode == 1) {
    float3 sourceHue = glossSurfaceSourceHueColor(meanR, meanG, meanB, u, idx);
    float shaped = pow(max(sourceLuma, 0.0), 0.85);
    float3 neutralBase = float3(0.10 + 0.52 * shaped,
                                0.10 + 0.50 * shaped,
                                0.11 + 0.46 * shaped);
    color = mixGloss3(neutralBase, sourceHue, 0.42);
  } else {
    float gray = 0.11 + 0.62 * pow(max(sourceLuma, 0.35 * bodyValue), 0.84);
    color = float3(gray * 0.98, gray * 0.985, gray);
  }
  float alpha = clamp((0.10 + 0.48 * structure) * bodyGain, 0.0, 0.68);
  return float4(clamp(color, 0.0, 1.0), alpha);
}

float4 glossSurfaceDisplayColorValues(float meanRValue,
                                      float meanGValue,
                                      float meanBValue,
                                      float carrierYValue,
                                      float carrierMaxValue,
                                      float carrierMinValue,
                                      float neutralityValue,
                                      float bodyValueIn,
                                      float positiveValue,
                                      float negativeValue,
                                      float boundaryValueIn,
                                      float congruenceValueIn,
                                      float confidenceValueIn,
                                      float signalValue,
                                      int colorMode,
                                      int debugMode,
                                      int diagnosticMode,
                                      float colorSaturation,
                                      float glossBodyOpacity,
                                      float glossHighlightOpacity,
                                      float glossLiftScale) {
  float base = clamp(bodyValueIn, 0.0, 1.0);
  float pos = clamp(positiveValue, 0.0, 1.0);
  float neg = clamp(negativeValue, 0.0, 1.0);
  if (debugMode != 0) {
    float scalar = 0.0;
    if (debugMode == 1) scalar = carrierMaxValue;
    else if (debugMode == 2) scalar = carrierYValue;
    else if (debugMode == 3) scalar = carrierMinValue;
    else if (debugMode == 4) scalar = neutralityValue;
    pos = clamp(scalar, 0.0, 1.0);
    neg = 0.0;
    base = clamp(scalar, 0.0, 1.0);
  }
  float confidenceValue = clamp(confidenceValueIn, 0.0, 1.0);
  float congruenceValue = clamp(congruenceValueIn, 0.0, 1.0);
  float boundaryValue = clamp(boundaryValueIn, 0.0, 1.0);
  float ambiguity = clamp(1.0 - confidenceValue, 0.0, 1.0);
  float signalScale = max(1.0, glossLiftScale);
  pos = clamp(pos * signalScale, 0.0, 1.0);
  neg = clamp(neg * signalScale, 0.0, 1.0);
  float positiveDisplay = smoothstep(0.035, 1.0, pos);
  float negativeDisplay = smoothstep(0.035, 1.0, neg);
  float signalPresence = max(positiveDisplay, negativeDisplay);
  float structureStrength = max(congruenceValue, boundaryValue);
  float3 color;
  if (colorMode == 1) {
    float sr = 0.0;
    float sg = 0.0;
    float sb = 0.0;
    mapDisplayColor(meanRValue, meanGValue, meanBValue, sr, sg, sb);
    applyDisplaySaturation(min(3.0, colorSaturation), sr, sg, sb);
    float3 sourceHue = float3(sr, sg, sb);
    float baseMix = clamp(glossBodyOpacity * (0.22 + 0.78 * confidenceValue) *
                              (0.86 - 0.22 * signalPresence),
                          0.0,
                          1.0);
    float shaped = pow(max(base, 0.0), 0.78);
    float3 neutralBase = float3(0.16 + 0.60 * shaped,
                                0.16 + 0.58 * shaped,
                                0.17 + 0.54 * shaped);
    color = mixGloss3(float3(0.03, 0.03, 0.04), mixGloss3(neutralBase, sourceHue, 0.68), baseMix);
    if (positiveDisplay > 0.0) {
      float3 warm = mixGloss3(sourceHue, float3(1.0, 0.95, 0.86), 0.54);
      color = mixGloss3(color,
                        warm,
                        clamp(glossHighlightOpacity * positiveDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
    if (negativeDisplay > 0.0) {
      float3 cool = mixGloss3(sourceHue, float3(0.08, 0.14, 0.24), 0.74);
      color = mixGloss3(color,
                        cool,
                        clamp(glossHighlightOpacity * negativeDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
  } else {
    float shaped = pow(max(base, 0.0), 0.78);
    float3 neutralBase = float3(0.16 + 0.64 * shaped,
                                0.16 + 0.64 * shaped,
                                0.17 + 0.60 * shaped);
    color = mixGloss3(float3(0.03, 0.03, 0.04),
                      neutralBase,
                      clamp(glossBodyOpacity * (0.22 + 0.78 * confidenceValue) *
                                (0.86 - 0.22 * signalPresence),
                            0.0,
                            1.0));
    if (positiveDisplay > 0.0) {
      color = mixGloss3(color,
                        float3(1.0, 0.89, 0.36),
                        clamp(glossHighlightOpacity * positiveDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
    if (negativeDisplay > 0.0) {
      color = mixGloss3(color,
                        float3(0.22, 0.76, 1.0),
                        clamp(glossHighlightOpacity * negativeDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
  }
  if (boundaryValue > 0.0) {
    color = mixGloss3(color, float3(0.98, 0.98, 0.94), clamp(0.10 + 0.26 * boundaryValue, 0.0, 0.34));
  }
  float alpha = clamp(glossBodyOpacity * (0.12 + 0.62 * confidenceValue) *
                          (0.82 - 0.18 * signalPresence) +
                          glossHighlightOpacity * signalPresence * (0.16 + 0.84 * structureStrength),
                      0.018,
                      1.0);
  if (diagnosticMode == 1) {
    float gray = 0.16 + 0.78 * confidenceValue;
    color = mixGloss3(color, float3(gray, gray, gray), 0.36);
    color = mixGloss3(color, float3(1.0, 1.0, 0.96), 0.10 * boundaryValue);
    alpha = clamp(alpha * (0.55 + 0.45 * confidenceValue) + 0.10 * confidenceValue, 0.018, 1.0);
  } else if (diagnosticMode == 2) {
    float gray = 0.12 + 0.74 * ambiguity;
    color = mixGloss3(color, float3(gray * 0.94, gray * 0.97, gray), 0.34);
    color = mixGloss3(color, float3(0.80, 0.90, 1.0), 0.10 * boundaryValue * ambiguity);
    alpha = clamp(alpha * (0.48 + 0.52 * ambiguity) + 0.08 * ambiguity, 0.018, 1.0);
  }
  (void)signalValue;
  return float4(clamp(color, 0.0, 1.0), alpha);
}

kernel void glossFieldSurfaceRenderKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                          const device float* meanR [[buffer(0)]],
                                          const device float* meanG [[buffer(1)]],
                                          const device float* meanB [[buffer(2)]],
                                          const device float* carrierY [[buffer(3)]],
                                          const device float* carrierMax [[buffer(4)]],
                                          const device float* carrierMin [[buffer(5)]],
                                          const device float* neutrality [[buffer(6)]],
                                          const device float* body [[buffer(7)]],
                                          const device float* positive [[buffer(8)]],
                                          const device float* negative [[buffer(9)]],
                                          const device float* boundary [[buffer(10)]],
                                          const device float* congruence [[buffer(11)]],
                                          const device float* confidence [[buffer(12)]],
                                          const device float* signal [[buffer(13)]],
                                          constant GlossFieldSurfaceUniforms& u [[buffer(14)]],
                                          uint2 gid [[thread_position_in_grid]]) {
  uint outWidth = outTexture.get_width();
  uint outHeight = outTexture.get_height();
  if (gid.x >= outWidth || gid.y >= outHeight ||
      u.gridWidth <= 0 || u.gridHeight <= 0 ||
      u.surfaceWidth <= 0 || u.surfaceHeight <= 0) return;
  float2 uv = float2((float(gid.x) + 0.5) / float(max(outWidth, 1u)),
                     (float(gid.y) + 0.5) / float(max(outHeight, 1u)));
  int x = clamp(int(floor(uv.x * float(u.gridWidth))), 0, u.gridWidth - 1);
  int yFromBottom = clamp(int(floor(uv.y * float(u.gridHeight))), 0, u.gridHeight - 1);
  int y = u.gridHeight - 1 - yFromBottom;
  uint idx = uint(y * u.gridWidth + x);
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, idx);
  float confidenceValue = clamp(confidence[idx], 0.0, 1.0);
  if (sourcePresence <= 0.01 && confidenceValue <= 0.01) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  float4 underlay = glossSurfaceUnderlayColor(meanR, meanG, meanB, body, confidence, u, idx);
  float4 display = glossSurfaceDisplayColorValues(meanR[idx],
                                                  meanG[idx],
                                                  meanB[idx],
                                                  carrierY[idx],
                                                  carrierMax[idx],
                                                  carrierMin[idx],
                                                  neutrality[idx],
                                                  body[idx],
                                                  positive[idx],
                                                  negative[idx],
                                                  boundary[idx],
                                                  congruence[idx],
                                                  confidence[idx],
                                                  signal[idx],
                                                  u.colorMode,
                                                  u.debugMode,
                                                  u.diagnosticMode,
                                                  u.colorSaturation,
                                                  u.glossBodyOpacity,
                                                  u.glossHighlightOpacity,
                                                  u.glossLiftScale);
  float4 color = overColor(underlay, display);
  outTexture.write(color, gid);
}

float4 multiplyProjectionMatrix(constant float* m, float4 v) {
  return float4(m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
                m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
                m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
                m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w);
}

bool glossProjectionCellScreenPosition(const device float* carrierY,
                                       const device float* carrierMax,
                                       const device float* carrierMin,
                                       const device float* neutrality,
                                       const device float* signal,
                                       constant GlossProjectionSurfaceUniforms& u,
                                       uint index,
                                       thread int* outPx,
                                       thread int* outPy,
                                       thread float* outNdcZ) {
  if (outPx == nullptr || outPy == nullptr || outNdcZ == nullptr) return false;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float aspect = clamp(u.sourceAspect, 0.25, 4.0);
  const float kMajorHalf = 1.22;
  float halfWidth = aspect >= 1.0 ? kMajorHalf : kMajorHalf * aspect;
  float halfDepth = aspect >= 1.0 ? kMajorHalf / aspect : kMajorHalf;
  float xNorm = (float(x) + 0.5) / float(u.gridWidth);
  float yNormInv = (float(y) + 0.5) / float(u.gridHeight);
  float xPos = -halfWidth + 2.0 * halfWidth * xNorm;
  float imageY = halfDepth - 2.0 * halfDepth * yNormInv;

  float signedValue = signal[index];
  if (u.debugMode == 1) {
    signedValue = clamp(carrierMax[index], 0.0, 1.0);
  } else if (u.debugMode == 2) {
    signedValue = clamp(carrierY[index], 0.0, 1.0);
  } else if (u.debugMode == 3) {
    signedValue = clamp(carrierMin[index], 0.0, 1.0);
  } else if (u.debugMode == 4) {
    signedValue = clamp(neutrality[index], 0.0, 1.0);
  }
  float zPos = u.debugMode == 0 ? signedValue : max(0.0, signedValue);

  float4 world = float4(xPos, imageY, zPos, 1.0);
  float4 eye = multiplyProjectionMatrix(u.modelView, world);
  float4 clip = multiplyProjectionMatrix(u.projection, eye);
  if (abs(clip.w) <= 1e-6) return false;
  float invW = 1.0 / clip.w;
  float ndcX = clip.x * invW;
  float ndcY = clip.y * invW;
  float ndcZ = clip.z * invW;
  if (ndcX < -1.12 || ndcX > 1.12 || ndcY < -1.12 || ndcY > 1.12 || ndcZ < -1.25 || ndcZ > 1.25) {
    return false;
  }
  *outPx = int(round((ndcX * 0.5 + 0.5) * float(u.surfaceWidth - 1)));
  int py = int(round((ndcY * 0.5 + 0.5) * float(u.surfaceHeight - 1)));
  *outPy = u.surfaceHeight - 1 - py;
  *outNdcZ = ndcZ;
  return true;
}

kernel void glossProjectionSurfaceSelectKernel(device atomic_uint* selection [[buffer(0)]],
                                               const device float* meanR [[buffer(1)]],
                                               const device float* meanG [[buffer(2)]],
                                               const device float* meanB [[buffer(3)]],
                                               const device float* carrierY [[buffer(4)]],
                                               const device float* carrierMax [[buffer(5)]],
                                               const device float* carrierMin [[buffer(6)]],
                                               const device float* neutrality [[buffer(7)]],
                                               const device float* confidence [[buffer(8)]],
                                               const device float* signal [[buffer(9)]],
                                               constant GlossProjectionSurfaceUniforms& u [[buffer(10)]],
                                               uint index [[thread_position_in_grid]]) {
  uint cellCount = uint(max(u.gridWidth * u.gridHeight, 0));
  if (index >= cellCount || u.surfaceWidth <= 0 || u.surfaceHeight <= 0) return;
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, index);
  float confidenceValue = clamp(confidence[index], 0.0, 1.0);
  if (sourcePresence <= 0.01 && confidenceValue <= 0.01) return;
  int px = 0;
  int py = 0;
  float ndcZ = 0.0;
  if (!glossProjectionCellScreenPosition(carrierY, carrierMax, carrierMin, neutrality, signal,
                                         u, index, &px, &py, &ndcZ)) {
    return;
  }
  float radius = clamp(u.pointRadiusPixels * (0.75 + 0.45 * confidenceValue), 1.0, 5.0);
  int r = int(ceil(radius));
  uint depthPriority = uint(clamp((1.25 - ndcZ) / 2.5, 0.0, 1.0) * 4095.0) + 1u;
  uint cellBits = min(index + 1u, 0x000FFFFFu);
  uint packed = (depthPriority << 20) | cellBits;
  for (int oy = -r; oy <= r; ++oy) {
    for (int ox = -r; ox <= r; ++ox) {
      float dist = sqrt(float(ox * ox + oy * oy));
      if (dist > radius) continue;
      int sx = px + ox;
      int sy = py + oy;
      if (sx < 0 || sx >= u.surfaceWidth || sy < 0 || sy >= u.surfaceHeight) continue;
      uint pixelIndex = uint(sy * u.surfaceWidth + sx);
      atomic_fetch_max_explicit(&selection[pixelIndex], packed, memory_order_relaxed);
    }
  }
}

kernel void glossProjectionSurfaceShadeKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                              device atomic_uint* selection [[buffer(0)]],
                                              const device float* meanR [[buffer(1)]],
                                              const device float* meanG [[buffer(2)]],
                                              const device float* meanB [[buffer(3)]],
                                              const device float* carrierY [[buffer(4)]],
                                              const device float* carrierMax [[buffer(5)]],
                                              const device float* carrierMin [[buffer(6)]],
                                              const device float* neutrality [[buffer(7)]],
                                              const device float* body [[buffer(8)]],
                                              const device float* positive [[buffer(9)]],
                                              const device float* negative [[buffer(10)]],
                                              const device float* boundary [[buffer(11)]],
                                              const device float* congruence [[buffer(12)]],
                                              const device float* confidence [[buffer(13)]],
                                              const device float* signal [[buffer(14)]],
                                              constant GlossProjectionSurfaceUniforms& u [[buffer(15)]],
                                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      u.surfaceWidth <= 0 || u.surfaceHeight <= 0) {
    return;
  }
  uint pixelIndex = gid.y * uint(u.surfaceWidth) + gid.x;
  uint packed = atomic_load_explicit(&selection[pixelIndex], memory_order_relaxed);
  uint cellIndex = packed & 0x000FFFFFu;
  if (cellIndex == 0u) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  uint index = cellIndex - 1u;
  uint cellCount = uint(max(u.gridWidth * u.gridHeight, 0));
  if (index >= cellCount) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  GlossFieldSurfaceUniforms fieldU;
  fieldU.gridWidth = u.gridWidth;
  fieldU.gridHeight = u.gridHeight;
  fieldU.surfaceWidth = u.surfaceWidth;
  fieldU.surfaceHeight = u.surfaceHeight;
  fieldU.algorithm = u.algorithm;
  fieldU.colorMode = u.colorMode;
  fieldU.debugMode = u.debugMode;
  fieldU.diagnosticMode = u.diagnosticMode;
  fieldU.colorSaturation = u.colorSaturation;
  fieldU.glossBodyOpacity = u.glossBodyOpacity;
  fieldU.glossHighlightOpacity = u.glossHighlightOpacity;
  fieldU.glossLiftScale = u.glossLiftScale;
  float4 display = glossSurfaceDisplayColorValues(meanR[index],
                                                  meanG[index],
                                                  meanB[index],
                                                  carrierY[index],
                                                  carrierMax[index],
                                                  carrierMin[index],
                                                  neutrality[index],
                                                  body[index],
                                                  positive[index],
                                                  negative[index],
                                                  boundary[index],
                                                  congruence[index],
                                                  confidence[index],
                                                  signal[index],
                                                  fieldU.colorMode,
                                                  fieldU.debugMode,
                                                  fieldU.diagnosticMode,
                                                  fieldU.colorSaturation,
                                                  fieldU.glossBodyOpacity,
                                                  fieldU.glossHighlightOpacity,
                                                  fieldU.glossLiftScale);
  outTexture.write(display, gid);
}

kernel void plotSurfaceClearKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                   constant PlotSurfaceClearUniforms& u [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
  outTexture.write(float4(u.r, u.g, u.b, u.a), gid);
}

kernel void sourceSignalSurfaceRenderKernel(texture2d<float, access::sample> sourceTexture [[texture(0)]],
                                            texture2d<float, access::write> outTexture [[texture(1)]],
                                            constant SourceSignalSurfaceUniforms& u [[buffer(0)]],
                                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
  float outputW = float(max(u.outputWidth, 1));
  float outputH = float(max(u.outputHeight, 1));
  float sourceW = float(max(u.sourceWidth, 1));
  float sourceH = float(max(u.sourceHeight, 1));
  float sourceAspect = sourceW / sourceH;
  float outputAspect = outputW / outputH;
  float imageW = outputW;
  float imageH = outputH;
  if (sourceAspect > outputAspect) {
    imageH = outputW / sourceAspect;
  } else {
    imageW = outputH * sourceAspect;
  }
  float offsetX = (outputW - imageW) * 0.5;
  float offsetY = (outputH - imageH) * 0.5;
  float px = float(gid.x) + 0.5;
  float py = float(gid.y) + 0.5;
  if (px < offsetX || px >= offsetX + imageW || py < offsetY || py >= offsetY + imageH) {
    outTexture.write(float4(u.backgroundR, u.backgroundG, u.backgroundB, 1.0), gid);
    return;
  }
  constexpr sampler sourceSampler(coord::normalized,
                                  address::clamp_to_edge,
                                  filter::linear);
  float2 uv = float2((px - offsetX) / max(imageW, 1.0),
                     (py - offsetY) / max(imageH, 1.0));
  float4 color = sourceTexture.sample(sourceSampler, uv);
  outTexture.write(float4(clamp(color.rgb, 0.0, 1.0), 1.0), gid);
}

vertex SurfaceCompositeVertexOut frameSurfaceCompositeVertex(uint vertexId [[vertex_id]],
                                                             constant SurfaceCompositeUniforms& u [[buffer(0)]]) {
  float2 corner;
  if (vertexId == 0) {
    corner = float2(0.0, 0.0);
  } else if (vertexId == 1) {
    corner = float2(1.0, 0.0);
  } else if (vertexId == 2) {
    corner = float2(0.0, 1.0);
  } else {
    corner = float2(1.0, 1.0);
  }
  float2 pixel = float2(u.dstX + corner.x * u.dstW,
                        u.dstY + corner.y * u.dstH);
  float2 ndc = float2((pixel.x / max(u.drawableW, 1.0)) * 2.0 - 1.0,
                      1.0 - (pixel.y / max(u.drawableH, 1.0)) * 2.0);
  SurfaceCompositeVertexOut out;
  out.position = float4(ndc, 0.0, 1.0);
  out.uv = corner;
  return out;
}

fragment float4 frameSurfaceCompositeFragment(SurfaceCompositeVertexOut in [[stage_in]],
                                              texture2d<float, access::sample> sourceTexture [[texture(0)]],
                                              constant SurfaceCompositeUniforms& u [[buffer(0)]]) {
  constexpr sampler surfaceSampler(coord::normalized,
                                   address::clamp_to_edge,
                                   filter::linear);
  float4 color = sourceTexture.sample(surfaceSampler, in.uv);
  color.a *= clamp(u.opacity, 0.0, 1.0);
  color.rgb *= clamp(u.opacity, 0.0, 1.0);
  return color;
}

vertex SurfaceCompositeVertexOut frameSolidRectVertex(uint vertexId [[vertex_id]],
                                                      constant FrameSolidRectUniforms& u [[buffer(0)]]) {
  float2 corner;
  if (vertexId == 0) {
    corner = float2(0.0, 0.0);
  } else if (vertexId == 1) {
    corner = float2(1.0, 0.0);
  } else if (vertexId == 2) {
    corner = float2(0.0, 1.0);
  } else {
    corner = float2(1.0, 1.0);
  }
  float2 pixel = float2(u.dstX + corner.x * u.dstW,
                        u.dstY + corner.y * u.dstH);
  float2 ndc = float2((pixel.x / max(u.drawableW, 1.0)) * 2.0 - 1.0,
                      1.0 - (pixel.y / max(u.drawableH, 1.0)) * 2.0);
  SurfaceCompositeVertexOut out;
  out.position = float4(ndc, 0.0, 1.0);
  out.uv = corner;
  return out;
}

fragment float4 frameSolidRectFragment(SurfaceCompositeVertexOut in [[stage_in]],
                                       constant FrameSolidRectUniforms& u [[buffer(0)]]) {
  return float4(u.r, u.g, u.b, clamp(u.a, 0.0, 1.0));
}

vertex FrameUiVectorVertexOut frameUiVectorVertex(uint vertexId [[vertex_id]],
                                                  constant FrameUiVectorVertexIn* vertices [[buffer(0)]],
                                                  constant FrameUiVectorUniforms& u [[buffer(1)]]) {
  FrameUiVectorVertexIn v = vertices[vertexId];
  float2 ndc = float2((v.x / max(u.drawableW, 1.0)) * 2.0 - 1.0,
                      1.0 - (v.y / max(u.drawableH, 1.0)) * 2.0);
  FrameUiVectorVertexOut out;
  out.position = float4(ndc, 0.0, 1.0);
  out.color = float4(v.r, v.g, v.b, v.a);
  return out;
}

fragment float4 frameUiVectorFragment(FrameUiVectorVertexOut in [[stage_in]]) {
  return float4(in.color.rgb, clamp(in.color.a, 0.0, 1.0));
}

float4 mulRasterModelViewColumnMajor(constant RasterPointSurfaceUniforms& u, float4 v) {
  return float4(u.modelView[0] * v.x + u.modelView[4] * v.y + u.modelView[8] * v.z + u.modelView[12] * v.w,
                u.modelView[1] * v.x + u.modelView[5] * v.y + u.modelView[9] * v.z + u.modelView[13] * v.w,
                u.modelView[2] * v.x + u.modelView[6] * v.y + u.modelView[10] * v.z + u.modelView[14] * v.w,
                u.modelView[3] * v.x + u.modelView[7] * v.y + u.modelView[11] * v.z + u.modelView[15] * v.w);
}

float4 mulRasterProjectionColumnMajor(constant RasterPointSurfaceUniforms& u, float4 v) {
  return float4(u.projection[0] * v.x + u.projection[4] * v.y + u.projection[8] * v.z + u.projection[12] * v.w,
                u.projection[1] * v.x + u.projection[5] * v.y + u.projection[9] * v.z + u.projection[13] * v.w,
                u.projection[2] * v.x + u.projection[6] * v.y + u.projection[10] * v.z + u.projection[14] * v.w,
                u.projection[3] * v.x + u.projection[7] * v.y + u.projection[11] * v.z + u.projection[15] * v.w);
}

vertex RasterPointSurfaceVertexOut rasterPointSurfaceVertex(
    uint vertexId [[vertex_id]],
    const device packed_float3* positions [[buffer(0)]],
    const device float4* colors [[buffer(1)]],
    constant RasterPointSurfaceUniforms& u [[buffer(2)]]) {
  packed_float3 p = positions[vertexId];
  float4 world = float4(p.x, p.y, p.z, 1.0);
  float4 view = mulRasterModelViewColumnMajor(u, world);
  float4 clip = mulRasterProjectionColumnMajor(u, view);
  RasterPointSurfaceVertexOut out;
  out.position = clip;
  out.color = colors[vertexId];
  out.pointSize = max(1.0, u.pointRadiusPixels * 2.0);
  return out;
}

fragment float4 rasterPointSurfaceFragment(RasterPointSurfaceVertexOut in [[stage_in]]) {
  return float4(in.color.rgb, clamp(in.color.a, 0.0, 1.0));
}

vertex FrameTextVertexOut frameTextVertex(uint vertexId [[vertex_id]],
                                          constant FrameTextVertexIn* vertices [[buffer(0)]],
                                          constant FrameTextUniforms& u [[buffer(1)]]) {
  FrameTextVertexIn v = vertices[vertexId];
  float2 ndc = float2((v.x / max(u.drawableW, 1.0)) * 2.0 - 1.0,
                      1.0 - (v.y / max(u.drawableH, 1.0)) * 2.0);
  FrameTextVertexOut out;
  out.position = float4(ndc, 0.0, 1.0);
  out.uv = float2(v.u, v.v);
  out.pixel = float2(v.x, v.y);
  return out;
}

fragment float4 frameTextFragment(FrameTextVertexOut in [[stage_in]],
                                  texture2d<float, access::sample> atlasTexture [[texture(0)]],
                                  constant FrameTextUniforms& u [[buffer(0)]]) {
  constexpr sampler atlasSampler(coord::normalized,
                                 address::clamp_to_edge,
                                 filter::linear);
  if (u.clipEnabled > 0.5) {
    const bool outside = in.pixel.x < u.clipX ||
                         in.pixel.y < u.clipY ||
                         in.pixel.x >= u.clipX + u.clipW ||
                         in.pixel.y >= u.clipY + u.clipH;
    if (outside) return float4(u.r, u.g, u.b, 0.0);
  }
  float alpha = atlasTexture.sample(atlasSampler, in.uv).r * clamp(u.a, 0.0, 1.0);
  return float4(u.r, u.g, u.b, alpha);
}
