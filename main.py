import copy
import sys
import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

pygame.init()


class Player:
    def __init__(self, pos):
        self.pos = pos
        self.ang = 0
        self.vision_ang = np.pi / 3
        self.max_sight = 300
        self.keys = {'w': 0, 'd': 0, 's': 0, 'a': 0}
        self.step_per_frame = 1
        self.rel = 0

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

            if map_check[0] < 0 or map_check[0] > (player_screen.width / block_size) - 1 or map_check[1] < 0 or \
                    map_check[1] > (player_screen.height / block_size) - 1:
                return None, None, False

            if distance > self.max_sight:
                return None, None, False

            if grid_map[map_check[1]][map_check[0]] == 9:
                col = [self.pos[0] + np.sin(ang) * distance, self.pos[1] + np.cos(ang) * distance]
                return distance, col, True

    def keyboard_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
                elif event.key == pygame.K_w:
                    self.keys['w'] = 1
                elif event.key == pygame.K_d:
                    self.keys['d'] = 1
                elif event.key == pygame.K_s:
                    self.keys['s'] = 1
                elif event.key == pygame.K_a:
                    self.keys['a'] = 1

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.keys['w'] = 0
                elif event.key == pygame.K_d:
                    self.keys['d'] = 0
                elif event.key == pygame.K_s:
                    self.keys['s'] = 0
                elif event.key == pygame.K_a:
                    self.keys['a'] = 0

        self.mouse_handler()
        self.move_player()

    def move_player(self):
        mid_angle = (self.ang + self.ang+self.vision_ang) / 2
        next_step_check = self.step_per_frame*10

        if self.keys['w']:   # Move Forward
            pdx = -next_step_check if np.sin(mid_angle) < 0 else next_step_check
            pdy = -next_step_check if np.cos(mid_angle) < 0 else next_step_check

            col_x = self.pos[0]+pdx
            col_y = self.pos[1]+pdy

            if grid_map[int((self.pos[1]) // block_size)][int(col_x) // block_size] < 5:
                self.pos[0] += self.step_per_frame * np.sin(mid_angle)

            if grid_map[int(col_y) // block_size][int((self.pos[0])) // block_size] < 5:
                self.pos[1] += self.step_per_frame * np.cos(mid_angle)

        if self.keys['d']:  # Strafe Left
            ang = np.pi / 2 + (self.ang + self.ang + self.vision_ang) / 2
            pdx = -next_step_check if np.sin(ang) < 0 else next_step_check
            pdy = -next_step_check if np.cos(ang) < 0 else next_step_check

            col_x = self.pos[0]+pdx
            col_y = self.pos[1]+pdy

            if grid_map[int(col_y) // block_size][int(self.pos[0]) // block_size] < 5:
                self.pos[1] += np.cos(ang) * self.step_per_frame
            if grid_map[int(self.pos[1]) // block_size][int(col_x) // block_size] < 5:
                self.pos[0] += np.sin(ang) * self.step_per_frame

        if self.keys['s']:   # Move Backward
            pdx = -next_step_check if np.sin(mid_angle) < 0 else next_step_check
            pdy = -next_step_check if np.cos(mid_angle) < 0 else next_step_check

            col_x = self.pos[0] - pdx
            col_y = self.pos[1] - pdy

            if grid_map[int((self.pos[1]) // block_size)][int(col_x // block_size)] < 5:
                self.pos[0] -= self.step_per_frame * np.sin(mid_angle)

            if grid_map[int(col_y // block_size)][int(self.pos[0] // block_size)] < 5:
                self.pos[1] -= self.step_per_frame * np.cos(mid_angle)

        if self.keys['a']:  # Strafe Right
            ang = -np.pi / 2 + (self.ang + self.ang + self.vision_ang) / 2

            pdx = -next_step_check if np.sin(ang) < 0 else next_step_check
            pdy = -next_step_check if np.cos(ang) < 0 else next_step_check

            col_x = self.pos[0]+pdx
            col_y = self.pos[1]+pdy

            if grid_map[int(col_y) // block_size][int(self.pos[0]) // block_size] < 5:
                self.pos[1] += np.cos(ang) * self.step_per_frame
            if grid_map[int(self.pos[1]) // block_size][int(col_x) // block_size] < 5:
                self.pos[0] += np.sin(ang) * self.step_per_frame

    def mouse_handler(self):
        mouse_x, _ = pygame.mouse.get_pos()
        if mouse_x <= 0 or mouse_x >= player_screen.width-10:
            pygame.mouse.set_pos(player_screen.width / 2, player_screen.height / 2)

        for i in range(2):
            if self.ang > np.pi:
                self.ang = -np.pi
            if self.ang < -np.pi:
                self.ang = np.pi

        mouse_x, _ = pygame.mouse.get_rel()

        self.ang += mouse_x / 150

        self.rel = mouse_x


class Screen:
    def __init__(self):
        self.width = 650
        self.height = 500
        self.test = 0

        self.screen = pygame.display.set_mode((self.width, self.height))

    def draw(self):
        self.screen.fill('white')

        self.draw_3dim()
        self.draw_mini_map(block_size / 5)

        for enemy in enemies:
            enemy.draw()

        pygame.display.flip()

    def draw_3dim(self):
        # print(np.arange(np.pi/4*3, np.pi, 0.005), np.arange(-np.pi, -np.pi/4*3, 0.005))
        # a = np.concatenate((np.arange(np.pi/4*3, np.pi, 0.005), np.arange(-np.pi, -np.pi/4*3, 0.005)))
        # const = a.shape[0]
        # print(const)

        step = 0.005
        if player.ang + player.vision_ang > np.pi:
            remainder = (player.ang + player.vision_ang) % np.pi
            arr1 = np.arange(player.ang, np.pi, step)
            arr2 = np.arange(-np.pi + step, -np.pi + remainder, step)
            arr = np.concatenate((arr1, arr2))
        else:
            arr = np.arange(player.ang, player.ang + player.vision_ang, step)

        const = arr.shape[0]

        for idx, ang in enumerate(arr):
            distance, col, valid = player.calculate_col(ang)
            if valid:
                mod_angle = ((player.ang + player.ang + player.vision_ang)/2)-ang
                draw_distance = distance * abs(np.cos(mod_angle))

                line_height = block_size*self.height/draw_distance

                for enemy in enemies:
                    if enemy.visible:
                        if enemy.dist_from_player > distance:
                            pygame.draw.rect(player_screen.screen, 'red',
                                             [enemy.x_norm * player_screen.width, 250, 10, 10])
                            # Draw Black % White Scene
                            pygame.draw.rect(self.screen, [np.sqrt(draw_distance / player.max_sight) * 255] * 3,
                                             [[(self.width / const) * idx, self.height/2 - line_height/2.5],
                                              [(self.width / const) + 1, line_height]])
                            # if self.x_norm is not None:

                    else:
                        # Draw Black % White Scene
                        pygame.draw.rect(self.screen, [np.sqrt(draw_distance / player.max_sight) * 255] * 3,
                                         [[(self.width / const) * idx, self.height/2 - line_height/2.5],
                                          [(self.width / const) + 1, line_height]])


                # Draw Wall Texture
                # col_side = 0
                # if round(col[0]) % block_size == 0:
                #     col_side = col[1]
                # elif round(col[1]) % block_size == 0:
                #     col_side = col[0]
                #
                # wall_col = wall_image.subsurface(
                #     (col_side % block_size) / block_size * (wall_image.get_width() - (self.width / const)), 0,
                #     (self.width / const)+1, wall_image.get_height())
                # wall_col = pygame.transform.scale(wall_col, (wall_col.get_width(), line_height))
                # self.screen.blit(wall_col, ((self.width / const) * idx, self.height / 2 - line_height / 2.5))

                # Do sky
                # print((self.width / const)*idx)
                # if self.height / 2 - line_height / 2.5 > 0:
                #     sky_col = sky_image.subsurface(
                #         (self.width / const)*idx, 0,
                #         (self.width / const) + 1, sky_image.get_height())
                #     # print(self.height / 2 - line_height / 2.5)
                # sky_col = pygame.transform.scale(sky_col, (sky_col.get_width(), self.height / 2 - line_height / 2.5))
                # self.screen.blit(sky_col, ((self.width / const)*idx, 0))

    def draw_mini_map(self, mini_block_size):
        pygame.draw.rect(self.screen, 'white',
                         [0, 0, mini_block_size * self.width / block_size,
                          mini_block_size * self.height / block_size])

        for idx_x in range(int(self.height / block_size) + 1):
            for idx_y in range(int(self.width / block_size) + 1):
                if idx_y == 0:
                    pygame.draw.line(self.screen, 'black', [0, mini_block_size * idx_x],
                                     [mini_block_size * self.width / block_size, mini_block_size * idx_x])
                if idx_x == 0:
                    pygame.draw.line(self.screen, 'black', [mini_block_size * idx_y, 0],
                                     [mini_block_size * idx_y, mini_block_size * self.height / block_size])
                if idx_x < int(self.height / block_size) and idx_y < int(self.width / block_size):
                    if grid_map[idx_x][idx_y] == 9:
                        pygame.draw.rect(self.screen, 'black',
                                         [idx_y * mini_block_size, idx_x * mini_block_size,
                                          mini_block_size, mini_block_size])

            mini_map_pos = [player.pos[0] / self.width * mini_block_size * self.width / block_size,
                            player.pos[1] / self.height * mini_block_size * self.height / block_size]
            pygame.draw.circle(self.screen, 'red', mini_map_pos, 2)

            pygame.draw.line(self.screen, 'red', mini_map_pos,
                             [mini_map_pos[0] + np.sin(player.ang) * 10,
                              mini_map_pos[1] + np.cos(player.ang) * 5], 1)
            pygame.draw.line(self.screen, 'red', mini_map_pos,
                             [mini_map_pos[0] + np.sin(player.ang + player.vision_ang) * 10,
                              mini_map_pos[1] + np.cos(player.ang + player.vision_ang) * 5], 1)

            # pygame.draw.circle(player_screen.screen, 'red', [enemy.pos[0] / block_size * mini_block_size,
            #                                                  enemy.pos[1] / block_size * mini_block_size], 2)
            #
            # x = player.pos[0] - enemy.pos[0]
            # y = player.pos[1] - enemy.pos[1]
            # a = np.arctan2(x, y)
            # t = [enemy.pos[0] / self.width * mini_block_size * self.width / block_size,
            #      enemy.pos[1] / self.height * mini_block_size * self.height / block_size]
            # pygame.draw.line(self.screen, 'black', t, [t[0] + np.sin(a) * 10,
            #                                            t[1] + np.cos(a) * 5], 1)


# class Enemy:
#
#     def __init__(self):
#         self.pos = [block_size*1+block_size/2, block_size*1+block_size/2]
#         self.last_move = [None, None]
#
#     def enemy_handler(self):
#         path = self.find_path()
#         block_to_move = path[-1]
#         block_to_move = [block_to_move[1], block_to_move[0]]
#         block_to_move = [block_to_move[0] - self.pos[0]//block_size, block_to_move[1] - self.pos[1]//block_size]
#
#         if self.last_move[0] is not None:
#             if [x*x for x in self.last_move] == [1, 0] and [x * x for x in block_to_move] == [0, 1]:
#                 if self.pos[0] % (block_size/2) == 0 and self.pos[0] % block_size != 0:
#                     pass
#                 else:
#                     block_to_move = copy.copy(self.last_move)
#
#             if [x*x for x in self.last_move] == [0, 1] and [x * x for x in block_to_move] == [1, 0]:
#                 if self.pos[1] % (block_size/2) == 0 and self.pos[1] % block_size != 0:
#                     pass
#                 else:
#                     block_to_move = copy.copy(self.last_move)
#
#         self.last_move = copy.copy(block_to_move)
#         self.move(block_to_move)
#
#     def move(self, block_to_move):
#         # print(block_to_move, self.pos[0] // block_size, self.pos[1] // block_size, block_to_move[0] - self.pos[0] // block_size, block_to_move[1] - self.pos[1] // block_size)
#         # block_to_move[0] - self.pos[0] // block_size, block_to_move[1] - self.pos[1] // block_size
#         self.pos[0] += block_to_move[0] / 2
#         self.pos[1] += block_to_move[1] / 2
#
#     def find_path(self):
#         start_pos = [int(self.pos[1]) // block_size, int(self.pos[0]) // block_size]
#         searching = True
#         queue = [start_pos]
#         visited_pos = []
#         path = []
#         priors = np.zeros([grid_map.shape[0], grid_map.shape[1], 2])
#         while searching:
#             curr_pos = queue.pop(0)
#
#             visited_pos.append(curr_pos)
#
#             # Get Neighbors Of curr_pos
#             neighbors = []
#             if grid_map[curr_pos[0] + 1][curr_pos[1]] == 0:  # Search Down
#                 neighbors.append([curr_pos[0] + 1, curr_pos[1]])
#             if grid_map[curr_pos[0] - 1][curr_pos[1]] == 0:  # Search Up
#                 neighbors.append([curr_pos[0] - 1, curr_pos[1]])
#             if grid_map[curr_pos[0]][curr_pos[1] - 1] == 0:  # Search Left
#                 neighbors.append([curr_pos[0], curr_pos[1] - 1])
#             if grid_map[curr_pos[0]][curr_pos[1] + 1] == 0:  # Search Right
#                 neighbors.append([curr_pos[0], curr_pos[1] + 1])
#
#             for neighbor in neighbors:
#                 # If neighbor Was Not Visited
#                 if neighbor not in queue and neighbor not in visited_pos:
#                     # If Found End Point
#                     if neighbor == [int(player.pos[1]) // block_size, int(player.pos[0]) // block_size]:
#                         searching = False
#                         # Find Path
#                         tmp = np.array(curr_pos)
#                         path = [tmp]
#
#                         while np.any(np.not_equal(path[-1], start_pos)):
#                             path.append(priors[int(tmp[0])][int(tmp[1])])
#                             tmp = priors[int(tmp[0])][int(tmp[1])]
#
#                         path = path[0:-1]
#                         break
#
#                     # Store History In priors
#                     priors[neighbor[0]][neighbor[1]] = curr_pos
#                     # Add The neighbor To queue
#                     queue.append(neighbor)
#
#         return path
#
#     def draw(self):
#         pass


class Enemy:
    def __init__(self, pos):
        self.pos = pos
        self.visible = False
        self.dist_from_player = None
        self.x_norm = None

    def calc_dist_from_player(self):
        return np.sqrt((self.pos[0] - player.pos[0])**2 + (self.pos[1] - player.pos[1])**2)

    def draw(self):
        x = self.pos[0] - player.pos[0]
        y = self.pos[1] - player.pos[1]
        ang_enemy_player = np.arctan2(x, y)

        self.x_norm = None
        if player.ang + player.vision_ang > np.pi:
            remainder = (player.ang + player.vision_ang) % np.pi
            if player.ang-0.5 < ang_enemy_player < np.pi or -np.pi < ang_enemy_player < -np.pi+remainder:
                self.visible = True
                if ang_enemy_player > 0:
                    self.x_norm = (ang_enemy_player - player.ang) / (player.ang + player.vision_ang - player.ang)
                else:
                    self.x_norm = (ang_enemy_player+2*np.pi - player.ang) / (player.ang + player.vision_ang - player.ang)
            else:
                self.visible = False
        else:
            if player.ang-0.5 < ang_enemy_player < player.ang+player.vision_ang:
                self.visible = True
                self.x_norm = (ang_enemy_player - player.ang) / (player.ang + player.vision_ang - player.ang)
            else:
                self.visible = False

        self.dist_from_player = self.calc_dist_from_player()
        # print(self.dist_from_player)



player_screen = Screen()
player = Player([175.0, 320.0])

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

fps = 60
fpsClock = pygame.time.Clock()

# image_path = 'wood weathered plank.png'
wall_image_path = 'texture.png'
# image_path = 'brick_texture.jpg'
# image_path = '04multia.jpg'
# image_path = 'black_white.png'

sky_image_path = 'sky.jpg'

wall_image = pygame.image.load(wall_image_path)
sky_image = pygame.image.load(sky_image_path)

block_size = 50

grid_map = np.array([[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                     [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                     [9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 0, 9],
                     [9, 0, 1, 0, 0, 0, 0, 0, 0, 9, 9, 0, 9],
                     [9, 0, 0, 0, 0, 9, 9, 0, 0, 9, 9, 0, 9],
                     [9, 0, 0, 0, 0, 0, 9, 0, 0, 9, 9, 0, 9],
                     [9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 9, 0, 9],
                     [9, 0, 0, 0, 0, 9, 9, 0, 0, 9, 9, 0, 9],
                     [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                     [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]])


def enemy_startup():
    arr = []
    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            if grid_map[i][j] == 1:
                arr.append(Enemy([j*block_size+block_size/2, i*block_size+block_size/2]))

    return arr


enemies = enemy_startup()


def main():
    while True:
        player.keyboard_handler()
        # enemy.enemy_handler()
        player_screen.draw()
        fpsClock.tick(fps)
        # print(fpsClock.get_fps())
        pygame.display.set_caption(str(fpsClock.get_fps()))


if __name__ == '__main__':
    main()
