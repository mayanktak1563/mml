# Installation Guide - YouTube AdBlocker Chrome Extension

## Quick Installation Steps

### 1. Download the Extension
- Clone or download this repository
- Locate the `youtube-adblocker` folder

### 2. Enable Developer Mode in Chrome
1. Open Chrome browser
2. Navigate to `chrome://extensions/`
3. Toggle "Developer mode" ON (top right corner)

### 3. Load the Extension
1. Click "Load unpacked" button
2. Select the `youtube-adblocker` folder
3. Click "Select Folder"

### 4. Verify Installation
- The YouTube AdBlocker icon should appear in your Chrome toolbar
- Click the icon to see the popup interface
- Navigate to YouTube to test ad blocking

## Testing the Extension

### Basic Functionality Test
1. **Go to YouTube homepage**
   - Check if sidebar ads are blocked
   - Look for the extension icon in toolbar

2. **Watch a video with ads**
   - Pre-roll ads should be skipped/blocked
   - Mid-roll ads should be handled
   - Banner/overlay ads should be hidden

3. **Use the popup interface**
   - Click extension icon
   - Toggle ad blocking on/off
   - Check ads blocked counter

### Troubleshooting

**Extension not loading:**
- Make sure you selected the correct folder (`youtube-adblocker`)
- Check that `manifest.json` is in the root of the selected folder
- Look for error messages in Chrome extensions page

**Ads still showing:**
- Refresh the YouTube page
- Make sure toggle is enabled in popup
- Some ads may take a moment to be detected

**Extension not working:**
- Check Chrome console for errors (F12 > Console)
- Verify you're on a YouTube page (*.youtube.com)
- Try disabling other ad blockers temporarily

## Files Overview

The extension consists of these key files:
- `manifest.json` - Extension configuration
- `popup.html` - User interface  
- `js/content.js` - Main ad blocking logic
- `js/background.js` - Extension background processes
- `css/` - Styling files
- `icons/` - Extension icons

## Next Steps

After installation:
1. Visit YouTube and test different page types
2. Check the ads blocked counter in the popup
3. Report any issues or ads that aren't blocked
4. Consider contributing improvements to the project

## Security Note

This extension:
- Only requests necessary permissions
- Operates locally in your browser
- Does not send data to external servers
- Source code is fully visible and auditable