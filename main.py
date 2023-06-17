import sys
import time
import numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


pygame.init()

fps = 60
fpsClock = pygame.time.Clock()

block_size = 64
width, height = 640, 480

screen = pygame.display.set_mode((width * 2, height))

# blocks = np.zeros([int(height/block_size), int(width/block_size)], dtype='int')
# print(blocks)
blocks = [[0]*int(width/block_size) for _ in range(int(height/block_size))]
# print(blocks)


blocks[5][5] = 1
blocks[5][7] = 1
blocks[5][9] = 1
blocks[5][3] = 1
blocks[5][1] = 1

blocks[3][5+1] = 1
blocks[3][7+1] = 1
blocks[3][0] = 1
# blocks[3][3+1] = 1
blocks[3][1+1] = 1

blocks[1][5] = 1
blocks[1][7] = 1
blocks[1][1] = 1
blocks[1][3] = 1
blocks[1][1] = 1


# blocks[4,5] = 1
# # blocks[4,6] = 1
# blocks[4,7] = 1
# blocks[4,8] = 1
# blocks[4,9] = 1
#
# blocks[8,7] = 1
#
# blocks[10,7] = 1
# blocks[8,7] = 1
# blocks[8,5] = 1
# blocks[2,5] = 1
# blocks[2,5] = 1
# blocks[10,3] = 1
# blocks[1,2] = 1
# blocks[8,5] = 1
# blocks[2,2] = 1
#
#
#
# blocks[3,1] = 1
# blocks[4,1] = 1
# # blocks[5,1] = 1
# blocks[6,1] = 1
# blocks[7,1] = 1
#
# blocks[10,2] = 1


class Player:
    def __init__(self, pos):
        self.pos = pos
        self.ang = [0, np.pi/3]
        self.max_sight = 300
        self.arrows = [0, 0, 0, 0, 0, 0]  # up right down left
        self.step_per_frame = 1

    def calculate_col(self, ang):
        end_point = [self.pos[0] + np.sin(ang)*self.max_sight, self.pos[1] + np.cos(ang)*self.max_sight]

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
            ray_unit_step_size[1] = np.sqrt(1 + (dx/dy)**2)

        # ray_unit_step_size = [np.sqrt(1 + (dy / dx) ** 2), np.sqrt(1 + (dx / dy) ** 2)]
        map_check = [int(self.pos[0] // block_size), int(self.pos[1] // block_size)]

        if dx > 0:  # look for blocks in right
            step[0] = 1
            ray_length[0] = ((self.pos[0] // block_size) * block_size + block_size - self.pos[0]) * ray_unit_step_size[0]
        else:  # look for blocks in left
            step[0] = -1
            ray_length[0] = (self.pos[0] - self.pos[0] // block_size * block_size) * ray_unit_step_size[0]

        if dy > 0:  # look for blocks bottom
            step[1] = 1
            ray_length[1] = ((self.pos[1] // block_size) * block_size + block_size - self.pos[1]) * ray_unit_step_size[1]
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

            if map_check[0] < 0 or map_check[0] > (width/block_size)-1 or map_check[1] < 0 or map_check[1] > (height/block_size)-1:
                return None, None, False

            if distance > self.max_sight:
                return None, None, False

            if blocks[map_check[1]][map_check[0]] == 1:
                col = [self.pos[0] + np.sin(ang) * distance, self.pos[1] + np.cos(ang) * distance]
                # distance /= np.cos(((self.ang[0]+self.ang[1])/2)-ang)
                return distance, col, True

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
                if self.arrows[4]:
                    self.ang[0] -= 0.02
                    self.ang[1] -= 0.02
                if self.arrows[5]:
                    self.ang[0] += 0.02
                    self.ang[1] += 0.02

            mouse_x, _ = pygame.mouse.get_pos()
            if mouse_x <= width or mouse_x >= 2*width-10:
                pygame.mouse.set_pos(width + width / 2, height / 2)

            mouse_x, _ = pygame.mouse.get_rel()

            self.ang[0] += mouse_x/100
            self.ang[1] += mouse_x/100

        mid_angle = (self.ang[0]+self.ang[1])/2
        if self.arrows[0]:
            x_col = int((self.pos[0] + np.sin(mid_angle) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] + np.cos(mid_angle) * self.step_per_frame) // block_size)
            if blocks[y_col][x_col] != 1:
                self.pos[0] += np.sin(mid_angle)*self.step_per_frame
                self.pos[1] += np.cos(mid_angle)*self.step_per_frame
        if self.arrows[1]:
            ang = np.pi / 2 + (self.ang[0] + self.ang[1]) / 2
            x_col = int((self.pos[0] + np.sin(mid_angle) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] + np.cos(mid_angle) * self.step_per_frame) // block_size)
            if blocks[y_col][x_col] != 1:
                self.pos[0] += np.sin(ang) * self.step_per_frame
                self.pos[1] += np.cos(ang) * self.step_per_frame
        if self.arrows[2]:
            x_col = int((self.pos[0] - np.sin(mid_angle) * self.step_per_frame) // block_size)
            y_col = int((self.pos[1] - np.cos(mid_angle) * self.step_per_frame) // block_size)
            if blocks[y_col][x_col] != 1:
                self.pos[0] -= np.sin(mid_angle) * self.step_per_frame
                self.pos[1] -= np.cos(mid_angle) * self.step_per_frame
        if self.arrows[3]:
            ang = -np.pi / 2 + (self.ang[0] + self.ang[1]) / 2
            self.pos[0] += np.sin(ang) * self.step_per_frame
            self.pos[1] += np.cos(ang) * self.step_per_frame


    def check_wall(self):
        pass

    def draw(self):
        pygame.draw.circle(screen, 'black', self.pos, 5)
        for i in range(2):
            x = self.pos[0] + np.sin(self.ang[i]) * self.max_sight
            y = self.pos[1] + np.cos(self.ang[i]) * self.max_sight
            pygame.draw.line(screen, 'red', self.pos, [x, y], 3)

        const = np.arange(player.ang[0], player.ang[1], 0.01).shape[0]
        for i, ang in enumerate(np.arange(player.ang[0], player.ang[1], 0.01)):
            distance, col, if_valid = player.calculate_col(ang)
            if if_valid:
                pygame.draw.circle(screen, 'red', col, 5)

                # half_ang = (player.ang[0]+player.ang[1])/2
                pygame.draw.rect(screen, [distance/player.max_sight*255]*3,  # *np.cos(half_ang)
                                 [[(width/const)*i + width, (height/player.max_sight)*distance/2],
                                  [(width/const)+1, height-2*(height/player.max_sight)*distance/2-10]])


def draw():
    screen.fill('white')

    for i in range(int(height/block_size)):
        for j in range(int(width/block_size)):
            if j == 0:
                pygame.draw.line(screen, 'black', [0, block_size * i], [width, block_size * i])
            if i == 0:
                pygame.draw.line(screen, 'black', [block_size * j, 0], [block_size * j, height])
            if blocks[i][j] == 1:
                pygame.draw.rect(screen, 'black', [j * block_size, i * block_size, block_size, block_size])

    player.draw()

    pygame.display.flip()


player = Player([50.0, 300.0])
# player = Player([64*4, 64*2])

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
