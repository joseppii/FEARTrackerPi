#include "pi5_drawing.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace pi5 {

// ============================================================================
// Simple 8x8 Bitmap Font Data
// Each character is 8 rows of 8 bits
// ============================================================================

static const uint8_t FONT_DATA[96][8] = {
    // Space (32)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // ! (33)
    {0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00},
    // " (34)
    {0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00},
    // # (35)
    {0x6C, 0xFE, 0x6C, 0x6C, 0xFE, 0x6C, 0x00, 0x00},
    // $ (36)
    {0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00},
    // % (37)
    {0x62, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x46, 0x00},
    // & (38)
    {0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00},
    // ' (39)
    {0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00},
    // ( (40)
    {0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00},
    // ) (41)
    {0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00},
    // * (42)
    {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00},
    // + (43)
    {0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00},
    // , (44)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30},
    // - (45)
    {0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00},
    // . (46)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00},
    // / (47)
    {0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00},
    // 0 (48)
    {0x3C, 0x66, 0x6E, 0x7E, 0x76, 0x66, 0x3C, 0x00},
    // 1 (49)
    {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00},
    // 2 (50)
    {0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00},
    // 3 (51)
    {0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00},
    // 4 (52)
    {0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00},
    // 5 (53)
    {0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00},
    // 6 (54)
    {0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00},
    // 7 (55)
    {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00},
    // 8 (56)
    {0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00},
    // 9 (57)
    {0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00},
    // : (58)
    {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00},
    // ; (59)
    {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00},
    // < (60)
    {0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00},
    // = (61)
    {0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00},
    // > (62)
    {0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00},
    // ? (63)
    {0x3C, 0x66, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00},
    // @ (64)
    {0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00},
    // A (65)
    {0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00},
    // B (66)
    {0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00},
    // C (67)
    {0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00},
    // D (68)
    {0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00},
    // E (69)
    {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00},
    // F (70)
    {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00},
    // G (71)
    {0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00},
    // H (72)
    {0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00},
    // I (73)
    {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00},
    // J (74)
    {0x3E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00},
    // K (75)
    {0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00},
    // L (76)
    {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00},
    // M (77)
    {0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0x00},
    // N (78)
    {0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00},
    // O (79)
    {0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00},
    // P (80)
    {0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00},
    // Q (81)
    {0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00},
    // R (82)
    {0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00},
    // S (83)
    {0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00},
    // T (84)
    {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00},
    // U (85)
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00},
    // V (86)
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00},
    // W (87)
    {0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00},
    // X (88)
    {0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00},
    // Y (89)
    {0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00},
    // Z (90)
    {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00},
    // [ (91)
    {0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00},
    // \ (92)
    {0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00},
    // ] (93)
    {0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00},
    // ^ (94)
    {0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00},
    // _ (95)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00},
    // ` (96)
    {0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00},
    // a (97)
    {0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00},
    // b (98)
    {0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00},
    // c (99)
    {0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00},
    // d (100)
    {0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00},
    // e (101)
    {0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00},
    // f (102)
    {0x1C, 0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00},
    // g (103)
    {0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C},
    // h (104)
    {0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00},
    // i (105)
    {0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00},
    // j (106)
    {0x0C, 0x00, 0x1C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38},
    // k (107)
    {0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00},
    // l (108)
    {0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00},
    // m (109)
    {0x00, 0x00, 0xEC, 0xFE, 0xD6, 0xC6, 0xC6, 0x00},
    // n (110)
    {0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00},
    // o (111)
    {0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00},
    // p (112)
    {0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60},
    // q (113)
    {0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06},
    // r (114)
    {0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00},
    // s (115)
    {0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00},
    // t (116)
    {0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x1C, 0x00},
    // u (117)
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00},
    // v (118)
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00},
    // w (119)
    {0x00, 0x00, 0xC6, 0xC6, 0xD6, 0xFE, 0x6C, 0x00},
    // x (120)
    {0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00},
    // y (121)
    {0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C},
    // z (122)
    {0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00},
    // { (123)
    {0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00},
    // | (124)
    {0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00},
    // } (125)
    {0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00},
    // ~ (126)
    {0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // DEL (127) - empty
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
};

// ============================================================================
// Helper: Set pixel with bounds checking
// ============================================================================

static inline void set_pixel(Pi5Image& img, int x, int y, const Pi5Scalar& color) {
    if (x < 0 || x >= img.width() || y < 0 || y >= img.height()) {
        return;
    }

    uint8_t* pixel = img.pixel_ptr(x, y);

    if (img.format() == Pi5PixelFormat::BGR8) {
        pixel[0] = static_cast<uint8_t>(std::clamp(color[0], 0.0f, 255.0f));
        pixel[1] = static_cast<uint8_t>(std::clamp(color[1], 0.0f, 255.0f));
        pixel[2] = static_cast<uint8_t>(std::clamp(color[2], 0.0f, 255.0f));
    } else if (img.format() == Pi5PixelFormat::RGB8) {
        pixel[0] = static_cast<uint8_t>(std::clamp(color[2], 0.0f, 255.0f));
        pixel[1] = static_cast<uint8_t>(std::clamp(color[1], 0.0f, 255.0f));
        pixel[2] = static_cast<uint8_t>(std::clamp(color[0], 0.0f, 255.0f));
    } else if (img.format() == Pi5PixelFormat::GRAY8) {
        // Use luminance formula
        float gray = 0.299f * color[2] + 0.587f * color[1] + 0.114f * color[0];
        pixel[0] = static_cast<uint8_t>(std::clamp(gray, 0.0f, 255.0f));
    }
}

// ============================================================================
// Basic Drawing Functions
// ============================================================================

void draw_rectangle(Pi5Image& img, const Pi5Rect& rect,
                    const Pi5Scalar& color, int thickness) {
    if (img.empty() || rect.empty()) return;

    thickness = std::clamp(thickness, 1, 10);

    // Draw horizontal lines (top and bottom)
    for (int t = 0; t < thickness; ++t) {
        // Top line
        int y = rect.y + t;
        for (int x = rect.x; x < rect.x + rect.width; ++x) {
            set_pixel(img, x, y, color);
        }
        // Bottom line
        y = rect.y + rect.height - 1 - t;
        for (int x = rect.x; x < rect.x + rect.width; ++x) {
            set_pixel(img, x, y, color);
        }
    }

    // Draw vertical lines (left and right)
    for (int t = 0; t < thickness; ++t) {
        // Left line
        int x = rect.x + t;
        for (int y = rect.y; y < rect.y + rect.height; ++y) {
            set_pixel(img, x, y, color);
        }
        // Right line
        x = rect.x + rect.width - 1 - t;
        for (int y = rect.y; y < rect.y + rect.height; ++y) {
            set_pixel(img, x, y, color);
        }
    }
}

void draw_filled_rectangle(Pi5Image& img, const Pi5Rect& rect,
                           const Pi5Scalar& color) {
    if (img.empty() || rect.empty()) return;

    for (int y = rect.y; y < rect.y + rect.height; ++y) {
        for (int x = rect.x; x < rect.x + rect.width; ++x) {
            set_pixel(img, x, y, color);
        }
    }
}

void draw_line(Pi5Image& img, const Pi5Point& p1, const Pi5Point& p2,
               const Pi5Scalar& color, int thickness) {
    if (img.empty()) return;

    // Bresenham's line algorithm with thickness
    int dx = std::abs(p2.x - p1.x);
    int dy = std::abs(p2.y - p1.y);
    int sx = (p1.x < p2.x) ? 1 : -1;
    int sy = (p1.y < p2.y) ? 1 : -1;
    int err = dx - dy;

    int x = p1.x;
    int y = p1.y;

    int half_thick = thickness / 2;

    while (true) {
        // Draw thick point
        for (int ty = -half_thick; ty <= half_thick; ++ty) {
            for (int tx = -half_thick; tx <= half_thick; ++tx) {
                set_pixel(img, x + tx, y + ty, color);
            }
        }

        if (x == p2.x && y == p2.y) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

void draw_circle(Pi5Image& img, const Pi5Point& center, int radius,
                 const Pi5Scalar& color, int thickness) {
    if (img.empty() || radius <= 0) return;

    // Midpoint circle algorithm
    for (int t = 0; t < thickness; ++t) {
        int r = radius - t;
        if (r <= 0) break;

        int x = r;
        int y = 0;
        int err = 0;

        while (x >= y) {
            // Draw 8 octants
            set_pixel(img, center.x + x, center.y + y, color);
            set_pixel(img, center.x + y, center.y + x, color);
            set_pixel(img, center.x - y, center.y + x, color);
            set_pixel(img, center.x - x, center.y + y, color);
            set_pixel(img, center.x - x, center.y - y, color);
            set_pixel(img, center.x - y, center.y - x, color);
            set_pixel(img, center.x + y, center.y - x, color);
            set_pixel(img, center.x + x, center.y - y, color);

            y++;
            err += 1 + 2 * y;
            if (2 * (err - x) + 1 > 0) {
                x--;
                err += 1 - 2 * x;
            }
        }
    }
}

void draw_filled_circle(Pi5Image& img, const Pi5Point& center, int radius,
                        const Pi5Scalar& color) {
    if (img.empty() || radius <= 0) return;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= radius * radius) {
                set_pixel(img, center.x + x, center.y + y, color);
            }
        }
    }
}

// ============================================================================
// Text Drawing
// ============================================================================

static void draw_char(Pi5Image& img, char c, int x, int y,
                      const Pi5Scalar& color, float scale) {
    if (c < 32 || c > 127) c = '?';

    int char_idx = c - 32;
    const uint8_t* glyph = FONT_DATA[char_idx];

    int scaled_size = static_cast<int>(8 * scale);
    if (scaled_size < 1) scaled_size = 1;

    for (int row = 0; row < 8; ++row) {
        uint8_t row_data = glyph[row];
        for (int col = 0; col < 8; ++col) {
            if (row_data & (0x80 >> col)) {
                // Draw scaled pixel
                int px = x + static_cast<int>(col * scale);
                int py = y + static_cast<int>(row * scale);

                for (int sy = 0; sy < static_cast<int>(scale); ++sy) {
                    for (int sx = 0; sx < static_cast<int>(scale); ++sx) {
                        set_pixel(img, px + sx, py + sy, color);
                    }
                }
            }
        }
    }
}

void draw_text(Pi5Image& img, const std::string& text,
               const Pi5Point& position, const Pi5Scalar& color,
               float scale) {
    if (img.empty() || text.empty()) return;

    int char_width = static_cast<int>(8 * scale);
    int x = position.x;

    for (char c : text) {
        draw_char(img, c, x, position.y, color, scale);
        x += char_width;
    }
}

Pi5Size get_text_size(const std::string& text, float scale) {
    int char_width = static_cast<int>(8 * scale);
    int char_height = static_cast<int>(8 * scale);
    return Pi5Size(static_cast<int>(text.length()) * char_width, char_height);
}

void draw_text_with_background(Pi5Image& img, const std::string& text,
                               const Pi5Point& position,
                               const Pi5Scalar& text_color,
                               const Pi5Scalar& bg_color,
                               float scale, int padding) {
    if (img.empty() || text.empty()) return;

    Pi5Size text_size = get_text_size(text, scale);

    Pi5Rect bg_rect(
        position.x - padding,
        position.y - padding,
        text_size.width + 2 * padding,
        text_size.height + 2 * padding
    );

    draw_filled_rectangle(img, bg_rect, bg_color);
    draw_text(img, text, position, text_color, scale);
}

// ============================================================================
// Composite Drawing
// ============================================================================

void draw_bbox_with_label(Pi5Image& img, const Pi5Rect& bbox,
                          const std::string& label,
                          const Pi5Scalar& color,
                          int thickness, float font_scale) {
    if (img.empty()) return;

    // Draw bounding box
    draw_rectangle(img, bbox, color, thickness);

    // Draw label above box if provided
    if (!label.empty()) {
        Pi5Size label_size = get_text_size(label, font_scale);

        // Position label above the box
        int label_x = bbox.x;
        int label_y = bbox.y - label_size.height - 4;

        // If label would be off-screen, put it inside the box
        if (label_y < 0) {
            label_y = bbox.y + 2;
        }

        // Draw background for label
        Pi5Scalar bg_color = color;  // Same as box color
        Pi5Scalar text_color(255, 255, 255);  // White text

        draw_text_with_background(img, label,
                                  Pi5Point(label_x, label_y),
                                  text_color, bg_color,
                                  font_scale, 2);
    }
}

void draw_tracking_info(Pi5Image& img, int frame_num, float fps,
                        float confidence, const Pi5Scalar& color) {
    if (img.empty()) return;

    char buffer[64];

    // Frame number
    snprintf(buffer, sizeof(buffer), "Frame: %d", frame_num);
    draw_text_with_background(img, buffer, Pi5Point(10, 10),
                              Pi5Scalar(255, 255, 255), color, 1.0f, 2);

    // FPS
    snprintf(buffer, sizeof(buffer), "FPS: %.1f", fps);
    draw_text_with_background(img, buffer, Pi5Point(10, 30),
                              Pi5Scalar(255, 255, 255), color, 1.0f, 2);

    // Confidence
    snprintf(buffer, sizeof(buffer), "Conf: %.2f", confidence);
    draw_text_with_background(img, buffer, Pi5Point(10, 50),
                              Pi5Scalar(255, 255, 255), color, 1.0f, 2);
}

void draw_crosshair(Pi5Image& img, const Pi5Point& center,
                    int size, const Pi5Scalar& color, int thickness) {
    if (img.empty()) return;

    // Horizontal line
    draw_line(img,
              Pi5Point(center.x - size, center.y),
              Pi5Point(center.x + size, center.y),
              color, thickness);

    // Vertical line
    draw_line(img,
              Pi5Point(center.x, center.y - size),
              Pi5Point(center.x, center.y + size),
              color, thickness);
}

}  // namespace pi5
