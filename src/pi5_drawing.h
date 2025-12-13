#pragma once

#include "pi5_image.h"
#include "pi5_types.h"
#include <string>

namespace pi5 {

// ============================================================================
// Basic Drawing Functions (replaces OpenCV drawing)
// ============================================================================

// Draw rectangle outline on image
// thickness: line thickness in pixels (1-10)
void draw_rectangle(Pi5Image& img, const Pi5Rect& rect,
                    const Pi5Scalar& color, int thickness = 1);

// Draw filled rectangle
void draw_filled_rectangle(Pi5Image& img, const Pi5Rect& rect,
                           const Pi5Scalar& color);

// Draw line between two points
void draw_line(Pi5Image& img, const Pi5Point& p1, const Pi5Point& p2,
               const Pi5Scalar& color, int thickness = 1);

// Draw circle outline
void draw_circle(Pi5Image& img, const Pi5Point& center, int radius,
                 const Pi5Scalar& color, int thickness = 1);

// Draw filled circle
void draw_filled_circle(Pi5Image& img, const Pi5Point& center, int radius,
                        const Pi5Scalar& color);

// ============================================================================
// Text Drawing (simple bitmap font)
// ============================================================================

// Draw text on image
// Uses a simple built-in bitmap font
// scale: text size multiplier (1.0 = 8 pixels high)
void draw_text(Pi5Image& img, const std::string& text,
               const Pi5Point& position, const Pi5Scalar& color,
               float scale = 1.0f);

// Get size of text string with given scale
Pi5Size get_text_size(const std::string& text, float scale = 1.0f);

// Draw text with background
// Draws a filled rectangle behind the text for better visibility
void draw_text_with_background(Pi5Image& img, const std::string& text,
                               const Pi5Point& position,
                               const Pi5Scalar& text_color,
                               const Pi5Scalar& bg_color,
                               float scale = 1.0f, int padding = 2);

// ============================================================================
// Composite Drawing (for tracking visualization)
// ============================================================================

// Draw bounding box with label
// label: text to display above the box (e.g., "conf: 0.95")
void draw_bbox_with_label(Pi5Image& img, const Pi5Rect& bbox,
                          const std::string& label,
                          const Pi5Scalar& color,
                          int thickness = 2, float font_scale = 1.0f);

// Draw tracking info overlay
// Draws frame number, FPS, and confidence in top-left corner
void draw_tracking_info(Pi5Image& img, int frame_num, float fps,
                        float confidence, const Pi5Scalar& color);

// Draw crosshair at center point
void draw_crosshair(Pi5Image& img, const Pi5Point& center,
                    int size, const Pi5Scalar& color, int thickness = 1);

}  // namespace pi5
