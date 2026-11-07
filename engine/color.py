"""Colour helpers used by the renderer."""

from __future__ import annotations

from .math3d import Vec3


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    """Convert HSV (all components in 0..1) to RGB floats."""
    hue %= 1.0
    saturation = clamp(saturation)
    value = clamp(value)

    sector = int(hue * 6.0) % 6
    fraction = hue * 6.0 - int(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    if sector == 0:
        return value, t, p
    if sector == 1:
        return q, value, p
    if sector == 2:
        return p, value, t
    if sector == 3:
        return p, q, value
    if sector == 4:
        return t, p, value
    return value, p, q


def hsv_to_hex(hue: float, saturation: float, value: float) -> str:
    red, green, blue = hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def normal_to_hex(normal: Vec3, brightness: float = 1.0) -> str:
    """Map a direction to a colour (used by the normals debug view)."""
    red = clamp((normal.x * 0.5 + 0.5) * brightness)
    green = clamp((normal.y * 0.5 + 0.5) * brightness)
    blue = clamp((normal.z * 0.5 + 0.5) * brightness)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
