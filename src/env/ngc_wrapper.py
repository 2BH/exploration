import gymnasium as gym
import copy
import numpy as np
import pygame
import pygame.freetype
import sys


class DiscreteParallelWorldWrapper(gym.Wrapper):
    MAX_INT = sys.maxsize 
    """This wrapper creates a parallel world of the given environment.

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env, disturbance_type="append", share_action=True, **kwargs):
        super().__init__(env)
        self._parallel_env = gym.make("BabyAI-BossLevelNoUnlock-v0", render_mode="rgb_array")
        assert disturbance_type in ["overwrite", "append", "random_overwrite", "black"], (
            "disturbance_type must be one of ['overwrite', 'append', 'random_overwrite', 'black']"
        )
        self.disturbance_type = disturbance_type
        image_observation_space = self.env.observation_space["image"]
        if self.disturbance_type == "overwrite" or self.disturbance_type == "random_overwrite":
            self.observation_space = image_observation_space
            self.overwrite_ratio = kwargs.get("overwrite_ratio", 0.1)
            assert 0 <= self.overwrite_ratio <= 1, "overwrite_ratio must be in [0, 1]"
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_observation_space.shape[0]*2, *image_observation_space.shape[1:]),
                dtype=image_observation_space.dtype
            )
        self.share_action = share_action
        if share_action:
            self.action_space = gym.spaces.Discrete(self.env.action_space.n*2)
        else:
            raise NotImplementedError("Share action is not implemented yet")
            self.action_space = gym.spaces.MultiDiscrete(
                np.array([self.env.action_space.n, self._parallel_env.action_space.n])
                )
        
        self._cur_actual_obs = None
        self._cur_noisy_obs = None
        self._cur_obs = None
        self._cur_mask = None
        
    def step(self, action):
        if self.share_action:
            if action < self.env.action_space.n:
                actual_obs, reward, terminated, truncated, info = self.env.step(action)
                noisy_obs, _, _, _, _ = self._parallel_env.step(6)
            else:
                actual_obs, reward, terminated, truncated, info = self.env.step(6)
                if self.disturbance_type == "black":
                    noisy_obs, _, _, _, _ = self._parallel_env.step(6)
                else:
                    noisy_obs, _, _, _, _ = self._parallel_env.step(action - self.env.action_space.n)
        else:
            actual_obs, reward, terminated, truncated, info = self.env.step(action[0])
            noisy_obs, _, _, _, _ = self._parallel_env.step(action[1])
        info["true_observation"] = copy.deepcopy(actual_obs)
        obs = self.get_obs(actual_obs["image"], noisy_obs["image"])
        self._cur_actual_obs = actual_obs["image"]
        self._cur_noisy_obs = noisy_obs["image"]
        self._cur_obs = obs
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        actual_obs, _ = self.env.reset(seed=seed)
        if seed is None:
            noisy_obs, _ = self._parallel_env.reset()
        else:
            noisy_obs, _ = self._parallel_env.reset(seed=self.MAX_INT-seed)
        
        obs = self.get_obs(actual_obs["image"], noisy_obs["image"])
        
        self._cur_actual_obs = actual_obs["image"]
        self._cur_noisy_obs = noisy_obs["image"]
        self._cur_obs = obs

        return obs, {"true_observation": actual_obs}
    
    def get_obs(self, obs, noisy_obs):
        # mask will has the same shape as obs
        if self.disturbance_type == "overwrite":
            mask = np.zeros_like(obs, dtype=bool)
            mask = mask.flatten()
            end_idx = int(self.overwrite_ratio * mask.size) // 3 * 3
            mask[:end_idx] = True
            mask = mask.reshape(obs.shape)
            obs[mask] = noisy_obs[mask]
            self._cur_mask = mask.reshape(obs.shape)
        elif self.disturbance_type == "append":
            obs = np.concatenate([obs, noisy_obs], axis=0)
            mask = np.zeros((obs.shape[0]//2, *obs.shape[1:]), dtype=bool)
            self._cur_mask = mask
        elif self.disturbance_type == "random_overwrite":
            mask = np.random.rand(*obs.shape[:2])
            mask = np.repeat(mask[:,:,None], obs.shape[2], axis=2) < self.overwrite_ratio
            obs[mask] = noisy_obs[mask]
            mask = mask.reshape(obs.shape)
            self._cur_mask = mask
        elif self.disturbance_type == "black":
            noisy_obs = np.zeros_like(noisy_obs)
            noisy_obs[:,:,1] = 5 # Set all color to be grey
            obs = np.concatenate([obs, noisy_obs], axis=0)
            mask = np.zeros_like(obs, dtype=bool)
            self._cur_mask = mask
        return obs

    def close(self):
        self.env.close()
        self._parallel_env.close()
    
    def render(self, mode="human", incl_pov=True):
        """
        Renders the current state of the environment.

        Parameters:
            mode (str): The rendering mode. Can be either "human" or "rgb_array". Default is "human".
            incl_pov (bool): Whether to include the agent's point of view image in the rendered output. Default is False.

        Returns:
            If mode is "rgb_array", returns the rendered image as a numpy array.
        """
        actual_img = self.env.get_frame(self.env.highlight, self.env.tile_size, self.env.agent_pov)
        noisy_img = self._parallel_env.get_frame(self._parallel_env.highlight, self._parallel_env.tile_size, self._parallel_env.agent_pov)
        if self.disturbance_type == "black":
            noisy_img = np.zeros_like(noisy_img) + 255
        img = np.concatenate([actual_img, noisy_img], axis=0)
        if incl_pov:
            target_width = noisy_img.shape[1]
            actual_pov_img = self.env.get_frame(self.env.highlight, self.env.tile_size, True)
            noisy_pov_img = self._parallel_env.get_frame(self._parallel_env.highlight, self._parallel_env.tile_size, True)
            if self.disturbance_type == "append":
                actual_pov_img = np.pad(actual_pov_img.transpose(1, 0, 2),
                                        ((0,0), (target_width//2-actual_pov_img.shape[0]//2, target_width//2-actual_pov_img.shape[1]//2), (0,0)),
                                        mode="constant", constant_values=255)[::-1,:, :]
                noisy_pov_img = np.pad(noisy_pov_img.transpose(1, 0, 2),
                                       ((0,0), (target_width//2-noisy_pov_img.shape[0]//2, target_width//2-noisy_pov_img.shape[1]//2), (0,0)),
                                       mode="constant", constant_values=255)[::-1,:, :]
                
                img = np.concatenate([img, actual_pov_img, noisy_pov_img], axis=0)
            elif self.disturbance_type == "black":
                actual_pov_img = np.pad(actual_pov_img.transpose(1, 0, 2),
                                        ((0,0), (target_width//2-actual_pov_img.shape[0]//2, target_width//2-actual_pov_img.shape[1]//2), (0,0)),
                                        mode="constant", constant_values=255)[::-1,:, :]
                noisy_pov_img = np.zeros_like(noisy_pov_img) + 255
                noisy_pov_img = np.pad(noisy_pov_img.transpose(1, 0, 2),
                                       ((0,0), (target_width//2-noisy_pov_img.shape[0]//2, target_width//2-noisy_pov_img.shape[1]//2), (0,0)),
                                       mode="constant", constant_values=255)[::-1,:, :]
                
                img = np.concatenate([img, actual_pov_img, noisy_pov_img], axis=0)
            else:
                # Resize the image to not be ignored
                resize_factor = actual_pov_img.shape[0] // self._cur_actual_obs.shape[0]
                mask = self._cur_mask.repeat(resize_factor, axis=0).repeat(resize_factor, axis=1)
                actual_pov_img[mask] = noisy_pov_img[mask]
                pov_img = actual_pov_img
                actual_pov_img = np.pad(pov_img.transpose(1, 0, 2), ((0,0), (target_width//2-pov_img.shape[0]//2, target_width//2-pov_img.shape[1]//2), (0,0)), mode="constant", constant_values=255)
                img = np.concatenate([img, actual_pov_img[::-1,:, :]], axis=0)

        if mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid, Upper: Actual, Lower: Noisy")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return img

        elif mode == "rgb_array":
            return img