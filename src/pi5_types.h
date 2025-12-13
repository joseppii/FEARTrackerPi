#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Pi5Size - replaces cv::Size
struct Pi5Size {
    int width;
    int height;

    Pi5Size() : width(0), height(0) {}
    Pi5Size(int w, int h) : width(w), height(h) {}

    int area() const { return width * height; }
    bool empty() const { return width <= 0 || height <= 0; }

    bool operator==(const Pi5Size& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const Pi5Size& other) const {
        return !(*this == other);
    }
};

// Pi5Point - replaces cv::Point
struct Pi5Point {
    int x;
    int y;

    Pi5Point() : x(0), y(0) {}
    Pi5Point(int x_, int y_) : x(x_), y(y_) {}

    bool operator==(const Pi5Point& other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Pi5Point& other) const {
        return !(*this == other);
    }
};

// Pi5Point2f - replaces cv::Point2f
struct Pi5Point2f {
    float x;
    float y;

    Pi5Point2f() : x(0.0f), y(0.0f) {}
    Pi5Point2f(float x_, float y_) : x(x_), y(y_) {}

    Pi5Point2f operator+(const Pi5Point2f& other) const {
        return Pi5Point2f(x + other.x, y + other.y);
    }
    Pi5Point2f operator-(const Pi5Point2f& other) const {
        return Pi5Point2f(x - other.x, y - other.y);
    }
    Pi5Point2f operator*(float scale) const {
        return Pi5Point2f(x * scale, y * scale);
    }

    bool operator==(const Pi5Point2f& other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Pi5Point2f& other) const {
        return !(*this == other);
    }
};

// Pi5Scalar - replaces cv::Scalar (for colors and mean values)
struct Pi5Scalar {
    float val[4];

    Pi5Scalar() : val{0.0f, 0.0f, 0.0f, 0.0f} {}
    Pi5Scalar(float v0) : val{v0, 0.0f, 0.0f, 0.0f} {}
    Pi5Scalar(float v0, float v1, float v2, float v3 = 0.0f) : val{v0, v1, v2, v3} {}

    float& operator[](int i) { return val[i]; }
    const float& operator[](int i) const { return val[i]; }

    bool operator==(const Pi5Scalar& other) const {
        return val[0] == other.val[0] && val[1] == other.val[1] &&
               val[2] == other.val[2] && val[3] == other.val[3];
    }
    bool operator!=(const Pi5Scalar& other) const {
        return !(*this == other);
    }
};

// Pi5Rect - replaces cv::Rect
struct Pi5Rect {
    int x;
    int y;
    int width;
    int height;

    Pi5Rect() : x(0), y(0), width(0), height(0) {}
    Pi5Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}

    int area() const { return width * height; }
    bool empty() const { return width <= 0 || height <= 0; }

    Pi5Size size() const { return Pi5Size(width, height); }

    // Top-left corner
    Pi5Point tl() const { return Pi5Point(x, y); }

    // Bottom-right corner
    Pi5Point br() const { return Pi5Point(x + width, y + height); }

    // Check if point is inside rectangle
    bool contains(const Pi5Point& pt) const {
        return pt.x >= x && pt.x < x + width && pt.y >= y && pt.y < y + height;
    }

    // Intersection operator (like cv::Rect & cv::Rect)
    Pi5Rect operator&(const Pi5Rect& other) const {
        int x1 = std::max(x, other.x);
        int y1 = std::max(y, other.y);
        int x2 = std::min(x + width, other.x + other.width);
        int y2 = std::min(y + height, other.y + other.height);

        if (x2 <= x1 || y2 <= y1) {
            return Pi5Rect();  // No intersection
        }
        return Pi5Rect(x1, y1, x2 - x1, y2 - y1);
    }

    // Union operator (like cv::Rect | cv::Rect)
    Pi5Rect operator|(const Pi5Rect& other) const {
        if (empty()) return other;
        if (other.empty()) return *this;

        int x1 = std::min(x, other.x);
        int y1 = std::min(y, other.y);
        int x2 = std::max(x + width, other.x + other.width);
        int y2 = std::max(y + height, other.y + other.height);

        return Pi5Rect(x1, y1, x2 - x1, y2 - y1);
    }

    bool operator==(const Pi5Rect& other) const {
        return x == other.x && y == other.y && width == other.width && height == other.height;
    }
    bool operator!=(const Pi5Rect& other) const {
        return !(*this == other);
    }
};

// Utility functions (replaces cv:: functions)
namespace pi5 {

// L2 norm of a point (replaces cv::norm)
inline float norm(const Pi5Point2f& p) {
    return std::sqrt(p.x * p.x + p.y * p.y);
}

// L2 distance between two points
inline float distance(const Pi5Point2f& p1, const Pi5Point2f& p2) {
    return norm(p1 - p2);
}

}  // namespace pi5
