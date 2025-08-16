# 3D Measurement Flutter App

Auto-capture 3D measurement app with configurable API endpoints.

## Features

✅ **Automatic Photo Capture** - 20 photos in 30 seconds
✅ **Real-Time Angle Guidance** - Gyroscope shows if you're holding phone correctly
✅ **Visual Orientation Indicator** - See exact pitch/roll angles live
✅ **Visual Guidance** - Step-by-step instructions
✅ **Configurable API** - Change server endpoint in settings
✅ **Image Metadata** - EXIF data preserved for accuracy
✅ **Error Handling** - Comprehensive error messages
✅ **Professional UI** - Material Design 3

## Quick Start

### Prerequisites
- Flutter SDK 3.0+
- Android device with USB debugging
- Server running (see backend setup)

### Installation

```bash
# Install dependencies
flutter pub get

# Run on device
flutter run
```

### Configuration

1. **Server IP**: Tap settings icon on home screen
2. **Update API URL**: Enter your server IP (e.g., `http://192.168.1.100:8000`)
3. **Test Connection**: App will verify server status

### Usage

1. **Launch app** → Check server status
2. **Start Measurement** → Opens camera
3. **Position object** → Follow on-screen instructions
4. **Auto-capture** → Walk slowly around object
5. **Wait for processing** → 60-90 seconds
6. **View results** → Measurements with confidence

## Image Capture Guidelines

### For Best Accuracy:
- 📸 **20 images** (auto-captured in 30 seconds)
- 🔄 **Complete 360° circle** around object
- 📏 **Maintain consistent distance** (50-70cm)
- 💡 **Good lighting** (no harsh shadows)
- 🎯 **70-80% overlap** between images
- 📱 **Hold steady** during capture

### What to Avoid:
- ❌ Moving too fast
- ❌ Changing distance
- ❌ Dim lighting
- ❌ Reflective objects
- ❌ Cluttered background

## Expected Results

### Before Calibration:
- Accuracy: ±5-15%
- Confidence: 20-40%

### After Calibration:
- Accuracy: ±2-5%
- Confidence: 60-80%

## Troubleshooting

### "Cannot connect to server"
- Check WiFi connection
- Verify server IP in settings
- Ensure server is running
- Check firewall (port 8000)

### "Camera permission denied"
- Go to Settings → Apps → Measurement App → Permissions
- Enable Camera permission

### "Low confidence results"
- Recapture with better lighting
- Ensure full 360° coverage
- Use manual calibration

## Build APK

```bash
# Debug build
flutter build apk

# Release build
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

## Technical Details

- **Language**: Dart 3.0+
- **Framework**: Flutter 3.0+
- **Camera**: High resolution (1024px)
- **Compression**: 85% JPEG quality
- **Upload**: Multipart form-data
- **Timeout**: 180 seconds

## Architecture

```
lib/
├── main.dart                    # Entry point
├── models/                      # Data models
├── services/                    # Business logic
├── screens/                     # UI screens
├── widgets/                     # Reusable widgets
└── utils/                       # Constants & helpers
```

## License

Same as parent project (see LICENSE file)

