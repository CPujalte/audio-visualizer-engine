# Audio Visualizer Utils

import numpy as np
from PIL import Image, ImageDraw
from collections.abc import Sequence
from typing import Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VisualizerElement(ABC):
    
    @abstractmethod
    def __call__(self, fr_width: int = 1920, fr_height: int = 1080) -> Image.Image:
        pass

@dataclass(frozen=True)
class VisElemLateralBars(VisualizerElement):
    """
    Generates a sequence of visualizer bars that react to the intensities provided in the array.
    Bool modes (non-exclusive):
    - Mirror: doubles the number of bars and places them symetrically.
    - Filp: reverses the order of the sequence of bars.
    Double (exclusive):
    - Mirror: mirrors the bars on the opposite side of the canvas.
    · Rotate: rotates the bars 180 degrees and places them on the opposite side of the canvas.
    """
    intensity_array:        np.ndarray
    default_color:          tuple[int, int, int, int]
    reaction_color:         tuple[int, int, int, int]
    side:                   Literal["bottom", "top", "left", "right"] = "bottom"
    mirror:                 bool = False
    flip:                   bool = False
    double:                 Literal["mirror", "rotate"] | None = None
    min_bar_height_coef:    float = 0
    max_bar_height_coef:    float = 1
    fr_margin:              int = 0
    bar_gap:                int = 0
    bar_corner_radius:      int = 0
    bar_border_color:       tuple[int, int, int, int] = (0, 0, 0, 0)
    bar_border_width:       int = 0

    def __call__(self, fr_width = 1920, fr_height = 1080) -> Image.Image:

        # Interchanges sides to always be working on the bottom part relative to the side, transposing the whole image at the end of the function
        working_fr_width: int = fr_width if self.side in ["top", "bottom"] else fr_height
        working_fr_height: int = fr_height if self.side in ["top", "bottom"] else fr_width

        if self.flip:
            working_intensity_array = self.intensity_array[::-1]
        else:
            working_intensity_array = self.intensity_array

        if self.mirror:
            working_intensity_array = np.concat([working_intensity_array, working_intensity_array[::-1]])

        # Auxiliary variables
        n_bars: int = len(working_intensity_array)
        min_bar_height: int = round(working_fr_height * self.min_bar_height_coef)
        max_bar_height: int = round(working_fr_height * self.max_bar_height_coef)
        bar_height_range: int = max_bar_height - min_bar_height

        avalible_bar_width: int = working_fr_width - 2*self.fr_margin - (n_bars-1)*self.bar_gap
        bar_width, remainder_width = divmod(avalible_bar_width, n_bars)
        width_start = self.fr_margin + remainder_width

        # Corner and color calculation and packing
        corner1_x_array = width_start + np.arange(n_bars) * (bar_width + self.bar_gap)
        corner2_x_array = corner1_x_array + bar_width
        corner1_y_array = working_fr_height - working_intensity_array * bar_height_range - min_bar_height
        corner2_y_array = np.full_like(working_intensity_array, working_fr_height + self.bar_corner_radius)

        corner_array = np.array([corner1_x_array, corner1_y_array, corner2_x_array, corner2_y_array])
        corner_array = corner_array.T

        temp_intensity_array = working_intensity_array[:, np.newaxis]
        color_array = np.multiply(self.default_color, (1-temp_intensity_array)) + np.multiply(self.reaction_color, temp_intensity_array)
        color_array = np.astype(np.round(color_array), np.uint8)

        # Draw of the first set of bars
        bar_img = Image.new("RGBA", (working_fr_width, working_fr_height), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(bar_img, "RGBA")
        for corners, color in zip(corner_array, color_array):
            bar_draw.rounded_rectangle(tuple(corners), self.bar_corner_radius, tuple(color), self.bar_border_color, self.bar_border_width)

        # Draw of the double set of bars if necessary
        if self.double is not None:
            match self.double:
                case "mirror":
                    corner1_x_double_array = corner1_x_array - 1
                    corner2_x_double_array = corner2_x_array - 1
                    corner1_y_double_array = working_fr_height - corner2_y_array - 1
                    corner2_y_double_array = working_fr_height - corner1_y_array - 1
                case "rotate":
                    corner1_x_double_array = working_fr_width - corner2_x_array - 1
                    corner2_x_double_array = working_fr_width - corner1_x_array - 1
                    corner1_y_double_array = working_fr_height - corner2_y_array - 1
                    corner2_y_double_array = working_fr_height - corner1_y_array - 1

            corner_double_array = np.array([corner1_x_double_array, corner1_y_double_array, corner2_x_double_array, corner2_y_double_array])
            corner_double_array = corner_double_array.T

            for corners, color in zip(corner_double_array, color_array):
                bar_draw.rounded_rectangle(tuple(corners), self.bar_corner_radius, tuple(color), self.bar_border_color, self.bar_border_width)

        # Transposing of image to match selected side
        match self.side:
            case "right":
                bar_img = bar_img.transpose(Image.ROTATE_90)
            case "top":
                bar_img = bar_img.transpose(Image.ROTATE_180)
            case "left":
                bar_img = bar_img.transpose(Image.ROTATE_270)

        return bar_img


@dataclass(frozen=True)
class VisElemRadialLines(VisualizerElement):
    """
    Renders audio visualizer lines radially around a point on a transparent background in RGBA.
    Min and Max line_height_coef between 0 and 1, with 1 reaching the border of the frame and 0 staying at the center.
    """
    intensity_array:            np.ndarray
    default_color:              tuple[int, int, int, int]
    reaction_color:             tuple[int, int, int, int]
    center_x:                   int | None = None
    center_y:                   int | None = None
    min_line_extension_coef:    float = 0.0
    max_line_extension_coef:    float = 1.0
    mirror:                     bool = False
    round_ends:                 bool = True
    line_width:                 int = 5

    def __call__(self, fr_width = 1920, fr_height = 1080) -> Image.Image:

        # Array duplication and transformation if symetrical parameter is set to true
        if self.mirror:
            working_intensity_array = np.concat([self.intensity_array, self.intensity_array[::-1]])
        else:
            working_intensity_array = self.intensity_array

        # Auxiliary variable calculation
        n_lines = len(working_intensity_array)
        radial_gap = 2*np.pi / n_lines
        min_line_extension = self.min_line_extension_coef * min(fr_height, fr_width) // 2
        max_line_extension = self.max_line_extension_coef * min(fr_height, fr_width) // 2
        line_extension_range = max_line_extension - min_line_extension
        working_center_x = fr_width//2 if self.center_x is None else self.center_x
        working_center_y = fr_height//2 if self.center_y is None else self.center_y

        # Coordinate conversion from polar to cartesian and calculation
        radius_array = min_line_extension + (line_extension_range * working_intensity_array)
        angle_array = np.pi - (np.linspace(0, 2*np.pi, n_lines, endpoint=False)) - radial_gap/2
        coord_x_array = np.round(radius_array * np.sin(angle_array)) + working_center_x
        coord_y_array = np.round(radius_array * np.cos(angle_array)) + working_center_y

        coord_array = np.array([np.full(n_lines, working_center_x), np.full(n_lines, working_center_y), coord_x_array, coord_y_array])
        coord_array = coord_array.T

        # Color array calculation
        working_intensity_array = working_intensity_array[:, np.newaxis]
        color_array = np.multiply(self.default_color, (1-working_intensity_array)) + np.multiply(self.reaction_color, working_intensity_array)
        color_array = np.astype(np.round(color_array), np.uint8)

        # Reordering of rendering for aesthetics if mirror=True, rendering bars closer to the bottom first and then going up radially and symmetrically
        if self.mirror:

            # Auxiliary variables
            half_n_lines = n_lines // 2
            first_half_array_mask = np.arange(half_n_lines-1, -1, -1)
            second_half_array_mask = np.arange(half_n_lines, n_lines)

            # Mask creation and application to coord and color arrays
            array_mask = np.column_stack((first_half_array_mask, second_half_array_mask)).ravel()
            coord_array = coord_array[array_mask]
            color_array = color_array[array_mask]

        # Blank canvas creation and line drawing
        line_img = Image.new("RGBA", (fr_width, fr_height), (0, 0, 0, 0))
        line_draw = ImageDraw.Draw(line_img, "RGBA")

        # End rounding circle coordinate calculation if necessary
        if self.round_ends:
            circle_radius: int = self.line_width // 2
            circle_corner1_x = coord_array[:, 2] - circle_radius
            circle_corner2_x = coord_array[:, 2] + circle_radius
            circle_corner1_y = coord_array[:, 3] - circle_radius
            circle_corner2_y = coord_array[:, 3] + circle_radius

            circle_coord_array = np.array([circle_corner1_x, circle_corner1_y, circle_corner2_x, circle_corner2_y])
            circle_coord_array = circle_coord_array.T
        
        for i, (coords, color) in enumerate(zip(coord_array, color_array)):
            line_draw.line(tuple(coords), tuple(color), self.line_width)
            if self.round_ends:
                line_draw.ellipse(tuple(circle_coord_array[i]), tuple(color))

        return line_img


@dataclass(frozen=True)
class VisElemRadialTriangles(VisualizerElement):
    """
    Renders audio visualizer triangles radially around a point on a transparent background in RGBA.
    Min and Max triang_height_coef between 0 and 1, with 1 reaching the border of the frame and 0 staying at the center.
    """
    intensity_array:            np.ndarray
    default_color:              tuple[int, int, int, int]
    reaction_color:             tuple[int, int, int, int]
    center_x:                   int | None = None
    center_y:                   int | None = None
    min_triang_extension_coef:  float = 0.0
    max_triang_extension_coef:  float = 1.0
    triang_angle_coef:          float = 1.0
    mirror:                     bool = False
    aa_scale:                   int = 2

    def __call__(self, fr_width = 1920, fr_height = 1080):

        # Anti Aliasing rescaling
        working_fr_width = fr_width * self.aa_scale
        working_fr_height = fr_height * self.aa_scale

        # Array duplication and transformation if symetrical parameter is set to true
        if self.mirror:
            working_intensity_array = np.concat([self.intensity_array, self.intensity_array[::-1]])
        else:
            working_intensity_array = self.intensity_array

        # Auxiliary variable calculation
        n_triang = len(working_intensity_array)
        radial_gap = 2*np.pi / n_triang
        inner_triang_angle = radial_gap * self.triang_angle_coef
        half_inner_angle = inner_triang_angle / 2
        min_triang_extension = self.min_triang_extension_coef * min(working_fr_height, working_fr_width) // 2
        max_triang_extension = self.max_triang_extension_coef * min(working_fr_height, working_fr_width) // 2
        triang_extension_range = max_triang_extension - min_triang_extension
        working_center_x = working_fr_width//2 if self.center_x is None else self.center_x
        working_center_y = working_fr_height//2 if self.center_y is None else self.center_y

        # Coordinate conversion from polar to cartesian and calculation of triangle corners
        radius_array = min_triang_extension + (triang_extension_range * working_intensity_array)
        center_angle_array = np.pi - (np.linspace(0, 2*np.pi, n_triang, endpoint=False)) - radial_gap/2

        corner1_x_array = np.round(radius_array * np.sin(center_angle_array - half_inner_angle)) + working_center_x
        corner2_x_array = np.round(radius_array * np.sin(center_angle_array + half_inner_angle)) + working_center_x
        corner1_y_array = np.round(radius_array * np.cos(center_angle_array - half_inner_angle)) + working_center_y
        corner2_y_array = np.round(radius_array * np.cos(center_angle_array + half_inner_angle)) + working_center_y

        corner_array = np.array([[np.full(n_triang, working_center_x), corner1_x_array, corner2_x_array],
                                 [np.full(n_triang, working_center_y), corner1_y_array, corner2_y_array]
                                 ])
        corner_array = corner_array.T

        # Color array calculation
        working_intensity_array = working_intensity_array[:, np.newaxis]
        color_array = np.multiply(self.default_color, (1-working_intensity_array)) + np.multiply(self.reaction_color, working_intensity_array)
        color_array = np.astype(np.round(color_array), np.uint8)

        # Image creation and triangle draw
        triang_img = Image.new("RGBA", (working_fr_width, working_fr_height), (0, 0, 0, 0))
        triang_draw = ImageDraw.Draw(triang_img, "RGBA")
        for coords, color in zip(corner_array, color_array):
            triang_draw.polygon((tuple(coords[0]), tuple(coords[1]), tuple(coords[2])), tuple(color))

        # Anti Aliasing downscaling
        if self.aa_scale != 1:
            triang_img = triang_img.resize((fr_width, fr_height), Image.Resampling.BOX)

        return triang_img


@dataclass(frozen=True)
class VisElemCircle(VisualizerElement):
    """ Renders a circle with radius proportional to an intensity parameter. """
    intensity:      float
    default_color:  tuple[int, int, int, int]
    reaction_color: tuple[int, int, int, int]
    min_radius:     int = 0
    max_radius:     int = 100
    center_x:       int | None = None
    center_y:       int | None = None
    border_width:   int = 0
    border_color:       tuple[int, int, int, int] = (0, 0, 0, 0)

    def __call__(self, fr_width = 1920, fr_height = 1080):

        # Auxiliary variables
        radius_range: int = self.max_radius - self.min_radius
        working_center_x: int = fr_width//2 if self.center_x is None else self.center_x
        working_center_y: int = fr_height//2 if self.center_y is None else self.center_y

        # Bounding box corner and color calculation
        radius = self.min_radius + radius_range * self.intensity
        corner1_x = working_center_x - radius
        corner2_x = working_center_x + radius
        corner1_y = working_center_y - radius
        corner2_y = working_center_y + radius
        corners = (corner1_x, corner1_y, corner2_x, corner2_y)

        color = np.array(self.default_color) * (1-self.intensity) + np.array(self.reaction_color) * self.intensity
        color = np.astype(np.round(color), np.uint8)

        # Circle drawing
        circle_img = Image.new("RGBA", (fr_width, fr_height))
        circle_draw = ImageDraw.Draw(circle_img, "RGBA")
        circle_draw.ellipse(corners, tuple(color), self.border_color, self.border_width)

        return circle_img

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VisElemGroup(VisualizerElement):
    """ Combines multiple VisualizerElements into a single RGBA Image. """
    visualizer_elements:  Sequence[VisualizerElement]

    def __call__(self, fr_width: int = 1920, fr_height: int = 1080) -> Image.Image:
        if self.visualizer_elements is None:
            return Image.new("RGBA", (fr_width, fr_height), (0, 0, 0, 0))

        main_img = Image.new("RGBA", (fr_width, fr_height), (0, 0, 0, 0))
        for element in self.visualizer_elements:
            main_img.alpha_composite(element(fr_width=fr_width, fr_height=fr_height))

        return main_img