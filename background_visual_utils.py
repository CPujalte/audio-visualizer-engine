# Background Visual Utils

import numpy as np
import math
from PIL import Image
from collections.abc import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MeshGenerator(ABC):

    @abstractmethod
    def __call__(self, fr_width: int = 1920, fr_height: int = 1080) -> np.ndarray:
        pass


@dataclass(frozen=True)
class MeshGenRadial(MeshGenerator):
    """ Returns a normalized [0, 1] matrix with values proportional to the distance to a selected center. """
    center_prop_x: float = 0.5
    center_prop_y: float = 0.5

    def __call__(self, fr_width = 1920, fr_height = 1080) -> np.ndarray:

        param_x = np.linspace(0, 1, fr_width, dtype=np.float32) - self.center_prop_x
        param_y = np.linspace(0, 1, fr_height, dtype=np.float32) - self.center_prop_y
        param_x = param_x / np.max(np.abs(param_x))
        param_y = param_y / np.max(np.abs(param_y))

        coord_x, coord_y = np.meshgrid(param_x, param_y)

        max_distance = 2**0.5
        param_z = np.sqrt(coord_x**2 + coord_y**2) / max_distance

        param_grid = param_z[:, :, np.newaxis]

        return param_grid


@dataclass(frozen=True)
class MeshGenLinearGradient(MeshGenerator):
    """ Returns a normalized [0, 1] matrix with values proportional to the distance to a line following a chosen angle and a displacement coeficient. """
    degree_angle: float = 0.0
    displacement_prop: float = 0.5

    def __call__(self, fr_width = 1920, fr_height = 1080) -> np.ndarray:

        offset_degree_angle = self.degree_angle - 90.0
        radian_angle: float = offset_degree_angle * np.pi / 180.0
        a_var: float = math.cos(radian_angle) * -1
        b_var: float = math.sin(radian_angle)

        displacement: float = 2 * self.displacement_prop - 1
        c_var: float = -1 * (abs(a_var) + abs(b_var)) * displacement

        param_x = np.linspace(-1, 1, fr_width, dtype=np.float32)
        param_y = np.linspace(-1, 1, fr_height, dtype=np.float32)
        coord_x, coord_y = np.meshgrid(param_x, param_y)

        param_z = np.abs(a_var*coord_x + b_var*coord_y + c_var)

        param_grid = param_z[:, :, np.newaxis]
        max_distance = param_grid.max()
        param_grid = param_grid / max_distance

        return param_grid


@dataclass(frozen=True)
class MeshGenRectangular(MeshGenerator):
    """ Returns a normalized [0, 1] matrix with values proportional to the distance to the closest border with selectable convergence point. """
    center_prop_x: float = 0.5
    center_prop_y: float = 0.5

    def __call__(self, fr_width = 1920, fr_height = 1080) -> np.ndarray:

        center_x: int = round(self.center_prop_x * fr_width)
        center_y: int = round(self.center_prop_y * fr_height)
        
        param_x_neg = np.linspace(1, 0, center_x, dtype=np.float32)
        param_x_pos = np.linspace(0, 1, fr_width-center_x, dtype=np.float32)
        param_y_neg = np.linspace(1, 0, center_y, dtype=np.float32)
        param_y_pos = np.linspace(0, 1, fr_height-center_y, dtype=np.float32)

        param_x = np.concat((param_x_neg, param_x_pos))
        param_y = np.concat((param_y_neg, param_y_pos))
        coord_x, coord_y = np.meshgrid(param_x, param_y)

        param_z = np.maximum(coord_x, coord_y)
        param_grid = param_z[:, :, np.newaxis]

        return param_grid

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MeshReaction(ABC):
    
    @abstractmethod
    def __call__(self, param_grid: np.ndarray, intensity: float) -> np.ndarray:
        pass


@dataclass(frozen=True)
class MeshReacExpand(MeshReaction):
    """ Applies an exponential transformation to a mesh proportionally to an intensity parameter. """
    max_exponent: float = 2.0

    def __call__(self, param_grid, intensity) -> np.ndarray:
        exponent = intensity * (self.max_exponent-1) + 1
        param_grid = param_grid**exponent

        return param_grid


@dataclass(frozen=True)
class MeshReacContract(MeshReaction):
    """ Applies an nroot transformation to a mesh proportionally to an intensity parameter. """
    max_root_index: float = 2.0

    def __call__(self, param_grid, intensity) -> np.ndarray:
        root_index = intensity * (self.max_root_index-1) + 1
        param_grid = param_grid**(1/root_index)

        return param_grid

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ColorReaction(ABC):

    @abstractmethod
    def __call__(self, color_grid: np.ndarray, intensity: float) -> np.ndarray:
        pass


@dataclass(frozen=True)
class ColorGamma(ColorReaction):
    """ Applies a gamma transformation to an RGBA 3d matrix proportionally to an intensity parameter. """
    min_gamma_exponent: float = 0.5

    def __call__(self, color_grid, intensity) -> np.ndarray:

        color_grid_rgb = color_grid[:, :, 0:3]
        color_grid_alpha = color_grid[:, :, 3, np.newaxis]

        gamma_exponent = (1-intensity) * (1-self.min_gamma_exponent) + self.min_gamma_exponent

        color_grid_rgb = color_grid_rgb / 255
        color_grid_rgb = color_grid_rgb ** gamma_exponent
        color_grid_rgb = color_grid_rgb * 255

        color_grid = np.concat((color_grid_rgb, color_grid_alpha), axis=2)

        return color_grid


@dataclass(frozen=True)
class ColorBrightness(ColorReaction):
    """ Applies a brightness transformation to an RGBA 3d matrix proportionally to an intensity parameter. """
    max_brightness_mult: float = 1.5

    def __call__(self, color_grid, intensity) -> np.ndarray:

        color_grid_rgb = color_grid[:, :, 0:3]
        color_grid_alpha = color_grid[:, :, 3, np.newaxis]

        brightness_mult = intensity * (self.max_brightness_mult-1) + 1

        color_grid_rgb = color_grid_rgb * brightness_mult
        color_grid_rgb = np.clip(color_grid_rgb, 0, 255)

        color_grid = np.concat((color_grid_rgb, color_grid_alpha), axis=2)

        return color_grid


@dataclass(frozen=True)
class ColorTint(ColorReaction):
    """ Applies a color tint (chosen color must be RGBA) transformation to an RGBA 3d matrix proportionally to an intensity parameter. """
    tint_color: tuple[int, int, int, int]
    max_tint_proportion: float = 0.5

    def __call__(self, color_grid, intensity) -> np.ndarray:

        tint_color_proportion = intensity * self.max_tint_proportion
        original_color_proportion = 1 - tint_color_proportion

        color_grid = color_grid * original_color_proportion + np.asarray(self.tint_color, dtype=np.float32) * tint_color_proportion

        return color_grid

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class DitheringEffect(ABC):

    @abstractmethod
    def __call__(self, color_grid: np.ndarray) -> np.ndarray:
        pass


@dataclass(frozen=True)
class DitheringUniform(DitheringEffect):
    """ Adds a noise layer following a uniform distribution to a color grid. """
    seed: int | None = None
    amplitude: float = 0.5

    def __call__(self, color_grid) -> np.ndarray:

        rng = np.random.default_rng(self.seed)

        color_grid_shape = color_grid.shape

        noise_shape = (color_grid_shape[0], color_grid_shape[1], 1)
        uniform_grid = rng.random(noise_shape, dtype=np.float32)

        noise_grid = (2*uniform_grid - 1.0) * self.amplitude

        color_grid[:, :, :3] = color_grid[:, :, :3] + noise_grid

        return color_grid


@dataclass(frozen=True)
class DitheringTriangular(DitheringEffect):
    """ Adds a noise layer following a triangular distribution to a color grid. """
    seed: int | None = None
    amplitude: float = 1

    def __call__(self, color_grid) -> np.ndarray:

        rng = np.random.default_rng(self.seed)

        color_grid_shape = color_grid.shape

        noise_shape = (color_grid_shape[0], color_grid_shape[1], 1)
        uniform_grid_1 = rng.random(noise_shape, dtype=np.float32)
        uniform_grid_2 = rng.random(noise_shape, dtype=np.float32)

        noise_grid = (uniform_grid_1 + uniform_grid_2 - 1.0) * self.amplitude

        color_grid[:, :, :3] = color_grid[:, :, :3] + noise_grid

        return color_grid


@dataclass(frozen=True)
class DitheringForceBanding(DitheringEffect):
    """ Forcefully creates visible banding by eliminating decimal precision. Effect grows proportional to banding_divisor. """
    banding_divisor: float = 2

    def __call__(self, color_grid) -> np.ndarray:

        color_grid[:, :, :3] = np.round(color_grid[:, :, :3] / self.banding_divisor) * self.banding_divisor

        return color_grid

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderBackground():
    """ Renders a single background Image in format RGBA. """
    main_color:     tuple[int, int, int, int]
    accent_color:   tuple[int, int, int, int]
    param_grid:     np.ndarray | None = None
    intensity:      float = 0.0
    mesh_generator: MeshGenerator = MeshGenRadial()
    mesh_reaction:  Sequence[MeshReaction] | None = None
    color_reaction: Sequence[ColorReaction] | None = None
    dithering:      DitheringEffect | None = None

    def __call__(self, fr_width: int = 1920, fr_height: int = 1080) -> Image.Image:

        # Grid creation in case of not provided
        intensity = float(self.intensity)
        if self.param_grid is None:
            working_grid = self.mesh_generator(fr_width=fr_width, fr_height=fr_height)
        else:
            if self.param_grid.size != (fr_width, fr_height):
                raise ValueError(f"Expected param_grid size was {(fr_width, fr_height)}, but size {self.param_grid.size} was recieved")
            working_grid = np.astype(self.param_grid, np.float32)

        # Mesh transformation applications
        if self.mesh_reaction is not None:
            for reaction in self.mesh_reaction:
                working_grid = reaction(working_grid, intensity)

        # Color mix between center and border colors
        working_main_color = np.array(self.main_color, dtype=np.float32)
        working_accent_color = np.array(self.accent_color, dtype=np.float32)
        color_grid = working_accent_color * working_grid + working_main_color * (1-working_grid)

        # Post processing color transformation applications
        if self.color_reaction is not None:
            for reaction in self.color_reaction:
                color_grid = reaction(color_grid, intensity)

        # Dithering application
        color_grid = self.dithering(color_grid) if self.dithering is not None else color_grid

        # Image creation and exportation
        color_grid = color_grid.clip(0, 255)
        color_grid = np.round(color_grid)
        color_grid = color_grid.astype(np.uint8)
        bg_img = Image.fromarray(color_grid)

        return bg_img