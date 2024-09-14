"""Visual Iterative Prompting functions.

Code to implement visual iterative prompting, an approach for querying VLMs.
"""

import copy
import dataclasses
import enum
import io
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import vip_utils


@enum.unique
class SupportedEmbodiments(str, enum.Enum):
    """Embodiments supported by VIP."""

    HF_DEMO = 'hf_demo'


@dataclasses.dataclass()
class Coordinate:
    """Coordinate with necessary information for visualizing annotation."""

    # 2D image coordinates for the target annotation
    xy: Tuple[int, int]
    # Color and style of the coord.
    color: Optional[float] = None
    radius: Optional[int] = None


@dataclasses.dataclass()
class Sample:
    """Single Sample mapping actions to Coordinates."""

    # 2D or 3D action
    action: np.ndarray
    # Coordinates for the main annotation
    coord: Coordinate
    # Coordinates for the text label
    text_coord: Coordinate
    # Label to display in the text label
    label: str


class VisualIterativePrompter:
    """Visual Iterative Prompting class."""

    def __init__(self, style, action_spec, embodiment):
        self.embodiment = embodiment
        self.style = style
        self.action_spec = action_spec
        self.fig_scale_size = None
        # image preparer
        # robot_to_image_canonical_coords

    def action_to_coord(self, action, image, arm_xy, do_project=False):
        """Converts candidate action to image coordinate."""
        return self.navigation_action_to_coord(
            action=action, image=image, center_xy=arm_xy, do_project=do_project
        )

    def navigation_action_to_coord(
        self, action, image, center_xy, do_project=False
    ):
        """Converts a ZXY or XY action to an image coordinate.

    Conversion is done based on style['focal_offset'] and action_spec['scale'].

    Args:
      action: z, y, x action in robot action space
      image: image
      center_xy: x, y in image space
      do_project: whether or not to project actions sampled outside the image to
        the edge of the image

    Returns:
      Dict coordinate with image x, y, arrow color, and circle radius.
    """
        if self.action_spec['scale'][0] == 0:  # no z dimension
            norm_action = [
                (action[d] - self.action_spec['loc'][d])
                / (2 * self.action_spec['scale'][d])
                for d in range(1, 3)
            ]
            norm_action_y, norm_action_x = norm_action
            norm_action_z = 0
        else:
            norm_action = [
                (action[d] - self.action_spec['loc'][d])
                / (2 * self.action_spec['scale'][d])
                for d in range(3)
            ]
            norm_action_z, norm_action_y, norm_action_x = norm_action
        focal_length = np.max([
            0.2,  # positive focal lengths only
            self.style['focal_offset']
            / (self.style['focal_offset'] + norm_action_z),
        ])
        image_x = center_xy[0] - (
            self.action_spec['action_to_coord'] * norm_action_x * focal_length
        )
        image_y = center_xy[1] - (
            self.action_spec['action_to_coord'] * norm_action_y * focal_length
        )
        if (
            vip_utils.coord_outside_image(
                Coordinate(xy=(image_x, image_y)), image, self.style['radius']
            )
            and do_project
        ):
            # project the arrow to the edge of the image if too large
            height, width, _ = image.shape
            max_x = (
                width - center_xy[0] - 2 * self.style['radius']
                if norm_action_x < 0
                else center_xy[0] - 2 * self.style['radius']
            )
            max_y = (
                height - center_xy[1] - 2 * self.style['radius']
                if norm_action_y < 0
                else center_xy[1] - 2 * self.style['radius']
            )
            rescale_ratio = min(
                np.abs([
                    max_x / (self.action_spec[
                                 'action_to_coord'] * norm_action_x),
                    max_y / (self.action_spec[
                                 'action_to_coord'] * norm_action_y),
                ])
            )
            image_x = (
                center_xy[0]
                - self.action_spec[
                    'action_to_coord'] * norm_action_x * rescale_ratio
            )
            image_y = (
                center_xy[1]
                - self.action_spec[
                    'action_to_coord'] * norm_action_y * rescale_ratio
            )

        return Coordinate(
            xy=(int(image_x), int(image_y)),
            color=0.1 * self.style['rgb_scale'],
            radius=int(self.style['radius']),
        )

    def sample_actions(
        self, image, arm_xy, loc, scale, true_action=None, max_itrs=1000
    ):
        """Sample actions from distribution.

    Args:
      image: image
      arm_xy: x, y in image space of arm
      loc: action distribution mean to sample from
      scale: action distribution variance to sample from
      true_action: action taken in demonstration if available
      max_itrs: number of tries to get a valid sample

    Returns:
      samples: Samples with associated actions, coords, text_coords, labels.
    """
        image = copy.deepcopy(image)

        samples = []
        actions = []
        coords = []
        text_coords = []
        labels = []

        # Keep track of oracle action if available.
        true_label = None
        if true_action is not None:
            actions.append(true_action)
            coord = self.action_to_coord(true_action, image, arm_xy)
            coords.append(coord)
            text_coords.append(
                vip_utils.coord_to_text_coord(coords[-1], arm_xy, coord.radius)
            )
            true_label = np.random.randint(self.style['num_samples'])
            # labels.append(str(true_label) + '*')
            labels.append(str(true_label))

        # Generate all action samples.
        for i in range(self.style['num_samples']):
            if i == true_label:
                continue
            itrs = 0

            # Generate action scaled appropriately.
            action = np.clip(
                np.random.normal(loc, scale),
                self.action_spec['min'],
                self.action_spec['max'],
            )

            # Convert sampled action to image coordinates.
            coord = self.action_to_coord(action, image, arm_xy)

            # Resample action if it results in invalid image annotation.
            adjusted_scale = np.array(scale)
            while (
                vip_utils.is_invalid_coord(
                    coord, coords, self.style['radius'] * 1.5, image
                )
                or vip_utils.coord_outside_image(coord, image,
                                                 self.style['radius'])
            ) and itrs < max_itrs:
                action = np.clip(
                    np.random.normal(loc, adjusted_scale),
                    self.action_spec['min'],
                    self.action_spec['max'],
                )
                coord = self.action_to_coord(action, image, arm_xy)
                itrs += 1
                # increase sampling range slightly if not finding a good sample
                adjusted_scale *= 1.1
                if itrs == max_itrs:
                    # If the final iteration results in invalid annotation, just clip
                    # to edge of image.
                    coord = self.action_to_coord(action, image, arm_xy,
                                                 do_project=True)

            # Compute image coordinates of text labels.
            radius = coord.radius
            text_coord = Coordinate(
                xy=vip_utils.coord_to_text_coord(coord, arm_xy, radius)
            )

            actions.append(action)
            coords.append(coord)
            text_coords.append(text_coord)
            labels.append(str(i))

        for i in range(len(actions)):
            sample = Sample(
                action=actions[i],
                coord=coords[i],
                text_coord=text_coords[i],
                label=str(i),
            )
            samples.append(sample)
        return samples

    def add_arrow_overlay_plt(self, image, samples, arm_xy):
        """Add arrows and circles to the image.

    Args:
      image: image
      samples: Samples to visualize.
      arm_xy: x, y image coordinates for EEF center.
      log_image: Boolean for whether to save to CNS.

    Returns:
      image: image with visual prompts.
    """
        # Add transparent arrows and circles
        overlay = image.copy()
        (original_image_height, original_image_width, _) = image.shape

        white = (
            self.style['rgb_scale'],
            self.style['rgb_scale'],
            self.style['rgb_scale'],
        )

        # Add arrows.
        for sample in samples:
            color = sample.coord.color
            cv2.arrowedLine(
                overlay, arm_xy, sample.coord.xy, color,
                self.style['thickness']
            )
        image = cv2.addWeighted(
            overlay,
            self.style['arrow_alpha'],
            image,
            1 - self.style['arrow_alpha'],
            0,
        )

        overlay = image.copy()
        # Add circles.
        for sample in samples:
            color = sample.coord.color
            radius = sample.coord.radius
            cv2.circle(
                overlay,
                sample.text_coord.xy,
                radius,
                color,
                self.style['thickness'] + 1,
            )
            cv2.circle(overlay, sample.text_coord.xy, radius, white, -1)
        image = cv2.addWeighted(
            overlay,
            self.style['circle_alpha'],
            image,
            1 - self.style['circle_alpha'],
            0,
        )

        dpi = plt.rcParams['figure.dpi']
        if self.fig_scale_size is None:
            # test saving a figure to decide size for text figure
            fig_size = (
            original_image_width / dpi, original_image_height / dpi)
            plt.subplots(1, figsize=fig_size)
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            fig = plt.gcf()
            fig.tight_layout(pad=0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            test_image = cv2.imdecode(
                np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.fig_scale_size = original_image_width / test_image.shape[1]

        # Add text to figure.
        fig_size = (
            self.fig_scale_size * original_image_width / dpi,
            self.fig_scale_size * original_image_height / dpi,
        )
        plt.subplots(1, figsize=fig_size)
        plt.imshow(image, cmap='binary')
        for sample in samples:
            plt.text(
                sample.text_coord.xy[0],
                sample.text_coord.xy[1],
                sample.label,
                ha='center',
                va='center',
                color='k',
                fontsize=self.style['fontsize'],
            )

        # Compile image.
        plt.axis('off')
        fig = plt.gcf()
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        image = cv2.imdecode(
            np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
        )

        image = cv2.resize(image,
                           (original_image_width, original_image_height))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image

    def fit(self, values, samples):
        """Fit a loc and scale to selected actions.

    Args:
      values: list of selected labels
      samples: list of all Samples

    Returns:
      loc: mean of selected distribution
      scale: variance of selected distribution
    """
        actions = [sample.action for sample in samples]
        labels = [sample.label for sample in samples]

        if not values:  # revert to initial distribution
            print('GPT failed to return integer arrows')
            loc = self.action_spec['loc']
            scale = self.action_spec['scale']
        elif len(values) == 1:  # single response, add a distribution over it
            index = \
            np.where([label == str(values[-1]) for label in labels])[0][0]
            action = actions[index]
            print('action', action)
            loc = action
            scale = self.action_spec['min_scale']
        else:  # fit distribution
            selected_actions = []
            for value in values:
                idx = np.where([label == str(value) for label in labels])[0][0]
                selected_actions.append(actions[idx])

            loc_scale = [
                scipy.stats.norm.fit(
                    [action[d] for action in selected_actions])
                for d in range(3)
            ]
            loc = [loc_scale[d][0] for d in range(3)]
            scale = np.clip(
                [loc_scale[d][1] for d in range(3)],
                self.action_spec['min_scale'],
                None,
            )

        return loc, scale
