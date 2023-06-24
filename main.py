import sys
import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

pygame.init()

fps = 60
fpsClock = pygame.time.Clock()

block_size = 50
width, height = 650, 500

screen = pygame.display.set_mode((width, height))

grid_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class Player:
    def __init__(self, pos):
        self.pos = pos
        self.ang = [0, np.pi / 3]
        self.max_sight = 300
        self.arrows = [0, 0, 0, 0, 0, 0]  # w d s a
        self.step_per_frame = 1

    def calculate_col(self, ang):
        end_point = [self.pos[0] + np.sin(ang) * self.max_sight, self.pos[1] + np.cos(ang) * self.max_sight]

        dx = end_point[0] - self.pos[0]
        dy = end_point[1] - self.pos[1]

        ray_length = [None, None]
        step = [0, 0]
        ray_unit_step_size = [None, None]

        if dx == 0:
            ray_unit_step_size[0] = 0
        else:
            ray_unit_step_size[0] = np.sqrt(1 + (dy / dx) ** 2)
        if dy == 0:
            ray_unit_step_size[1] = 0
        else:
            ray_unit_step_size[1] = np.sqrt(1 + (dx / dy) ** 2)

        # ray_unit_step_size = [np.sqrt(1 + (dy / dx) ** 2), np.sqrt(1 + (dx / dy) ** 2)]
        map_check = [int(self.pos[0] // block_size), int(self.pos[1] // block_size)]

        if dx > 0:  # look for blocks in right
            step[0] = 1
            ray_length[0] = ((self.pos[0] // block_size) * block_size + block_size - self.pos[0]) * ray_unit_step_size[
                0]
        else:  # look for blocks in left
            step[0] = -1
            ray_length[0] = (self.pos[0] - self.pos[0] // block_size * block_size) * ray_unit_step_size[0]

        if dy > 0:  # look for blocks bottom
            step[1] = 1
            ray_length[1] = ((self.pos[1] // block_size) * block_size + block_size - self.pos[1]) * ray_unit_step_size[
                1]
        else:  # look for blocks up
            step[1] = -1
            ray_length[1] = (self.pos[1] - self.pos[1] // block_size * block_size) * ray_unit_step_size[1]

        while True:
            if dx == 0:
                map_check[1] += step[1]
                distance = ray_length[1]
                ray_length[1] += ray_unit_step_size[1] * block_size
            else:
                if ray_length[0] < ray_length[1]:
                    map_check[0] += step[0]
                    distance = ray_length[0]
                    ray_length[0] += ray_unit_step_size[0] * block_size
                else:
                    map_check[1] += step[1]
                    distance = ray_length[1]
                    ray_length[1] += ray_unit_step_size[1] * block_size

            if map_check[0] < 0 or map_check[0] > (width / block_size) - 1 or map_check[1] < 0 or map_check[1] > (
                    height / block_size) - 1:
                return None, False

            if distance > self.max_sight:
                return None, False

            if grid_map[map_check[1]][map_check[0]] == 1:
                return distance, True

    def keyboard_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
                if event.key == pygame.K_w:
                    self.arrows[0] = 1
                if event.key == pygame.K_d:
                    self.arrows[1] = 1
                if event.key == pygame.K_s:
                    self.arrows[2] = 1
                if event.key == pygame.K_a:
                    self.arrows[3] = 1
                if event.key == pygame.K_LEFT:
                    self.arrows[4] = 1
                if event.key == pygame.K_RIGHT:
                    self.arrows[5] = 1

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.arrows[0] = 0
                if event.key == pygame.K_d:
                    self.arrows[1] = 0
                if event.key == pygame.K_s:
                    self.arrows[2] = 0
                if event.key == pygame.K_a:
                    self.arrows[3] = 0
                if event.key == pygame.K_LEFT:
                    self.arrows[4] = 0
                if event.key == pygame.K_RIGHT:
                    self.arrows[5] = 0

            mouse_x, _ = pygame.mouse.get_pos()
            if mouse_x <= width or mouse_x >= 2 * width - 10:
                pygame.mouse.set_pos(width + width / 2, height / 2)

            mouse_x, _ = pygame.mouse.get_rel()

            self.ang[0] += mouse_x / 150
            self.ang[1] += mouse_x / 150

        # Move In Direction
        mid_angle = (self.ang[0] + self.ang[1]) / 2
        if self.arrows[0]:   # Move Forward
            x_col = int((self.pos[0] + np.sin(mid_angle) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] + np.cos(mid_angle) * self.step_per_frame) // block_size)
            if grid_map[y_col][x_col] != 1:
                self.pos[0] += np.sin(mid_angle) * self.step_per_frame
                self.pos[1] += np.cos(mid_angle) * self.step_per_frame
        if self.arrows[1]:  # Straight Left
            ang = np.pi / 2 + (self.ang[0] + self.ang[1]) / 2
            x_col = int((self.pos[0] + np.sin(ang) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] + np.cos(ang) * self.step_per_frame) // block_size)
            if grid_map[y_col][x_col] != 1:
                self.pos[0] += np.sin(ang) * self.step_per_frame
                self.pos[1] += np.cos(ang) * self.step_per_frame
        if self.arrows[2]:   # Move Backward
            x_col = int((self.pos[0] - np.sin(mid_angle) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] - np.cos(mid_angle) * self.step_per_frame) // block_size)
            if grid_map[y_col][x_col] != 1:
                self.pos[0] -= np.sin(mid_angle) * self.step_per_frame
                self.pos[1] -= np.cos(mid_angle) * self.step_per_frame
        if self.arrows[3]:  # Straight Right
            ang = np.pi / 2 + (self.ang[0] + self.ang[1]) / 2
            x_col = int((self.pos[0] - np.sin(ang) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] - np.cos(ang) * self.step_per_frame) // block_size)
            if grid_map[y_col][x_col] != 1:
                self.pos[0] -= np.sin(ang) * self.step_per_frame
                self.pos[1] -= np.cos(ang) * self.step_per_frame

        # Rotate The Player
        if self.arrows[4]:
            self.ang[0] -= 0.02
            self.ang[1] -= 0.02
        if self.arrows[5]:
            self.ang[0] += 0.02
            self.ang[1] += 0.02

    def draw_3dim(self):
        const = np.arange(self.ang[0], self.ang[1], 0.005).shape[0]
        for idx, ang in enumerate(np.arange(self.ang[0], self.ang[1], 0.005)):
            distance, if_valid = self.calculate_col(ang)
            if if_valid:
                mod_angle = ((self.ang[0]+self.ang[1])/2)-ang
                distance *= np.cos(mod_angle)

                line_height = block_size*height/distance
                pygame.draw.rect(screen, [np.sqrt(distance / self.max_sight) * 255] * 3,
                                 [[(width / const) * idx, height/2 - line_height/2.5],
                                  [(width / const) + 1, line_height]])


def draw_mini_map(mini_block_size):
    pygame.draw.rect(screen, 'white',
                     [0, 0, mini_block_size * width / block_size, mini_block_size * height / block_size])

    for idx_x in range(int(height / block_size) + 1):
        for idx_y in range(int(width / block_size) + 1):
            if idx_y == 0:
                pygame.draw.line(screen, 'black', [0, mini_block_size * idx_x],
                                 [mini_block_size * width / block_size, mini_block_size * idx_x])
            if idx_x == 0:
                pygame.draw.line(screen, 'black', [mini_block_size * idx_y, 0],
                                 [mini_block_size * idx_y, mini_block_size * height / block_size])
            if idx_x < int(height / block_size) and idx_y < int(width / block_size):
                if grid_map[idx_x][idx_y] == 1:
                    pygame.draw.rect(screen, 'black', [idx_y * mini_block_size, idx_x * mini_block_size,
                                                       mini_block_size, mini_block_size])

        mini_map_pos = [player.pos[0] / width * mini_block_size * width / block_size,
                        player.pos[1] / height * mini_block_size * height / block_size]
        pygame.draw.circle(screen, 'red', mini_map_pos, 2)

        pygame.draw.line(screen, 'red', mini_map_pos,
                         [mini_map_pos[0] + np.sin(player.ang[0]) * 10, mini_map_pos[1] + np.cos(player.ang[0]) * 5], 1)
        pygame.draw.line(screen, 'red', mini_map_pos,
                         [mini_map_pos[0] + np.sin(player.ang[1]) * 10, mini_map_pos[1] + np.cos(player.ang[1]) * 5], 1)


def draw():
    screen.fill('white')

    player.draw_3dim()
    draw_mini_map(block_size/5)

    pygame.display.flip()


player = Player([200.0, 320.0])

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)


def main():
    while True:
        player.keyboard_handler()
        draw()
        fpsClock.tick(fps)
        # print(fpsClock.get_fps())


if __name__ == '__main__':
    main()
