# 使用 游戏手柄 发送指令
import pygame
import time

pygame.init()
pygame.joystick.init()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:       # 如果点击关闭键
            done = True

        joystick_count = pygame.joystick.get_count()

        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            axes = joystick.get_numaxes()
            print("="*60)
            for j in range(axes):
                axis = joystick.get_axis(j)
                print("轴", j, "的值为：", axis)