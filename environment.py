from typing import Optional

import gym
import numpy as np
import pygame
from Box2D import *
import random
import math
from gym import spaces

width, height = 900, 600
# width is 900 pixel,height is 600 pixel.
# Create a Box2D physical world.
# Pygame is measured in pixels, Box_2d is measured in meters.
PPM = 20.0
# The 2d-world is actually 45 meters and 30 meters.
friction = 3
step_size = 0.5
# Every 0.5 meter, ground undulates one times.
body_h = 0.3
# Agent height is 0.3.
body_w = 1.3
# Agent weight is 1.3.
leg_h = 1
# Leg height of agent is 1.
leg_w = 0.20
# Leg  weight of agent is 0.25.
# At present, this part is used to test whether the world can interact with agent.
platform_length = 20
# The journey agent need to walk by itself.
init_height = height / 4 / PPM * 3
# Ground height, with the upper left corner as the origin,
# meaning that the height is one fourth.

class ContactDetector(b2ContactListener):
    def __init__(self, env):
        # Define contact detector, inherited from Box2D's contact listener.
        b2ContactListener.__init__(self)
        # Call the parent class initialization method.
        self.env = env
        # Save references to environment objects.

    def BeginContact(self, contact):
        # If the body, head, or tail touches another object, the game ends.
        if (
                 contact.fixtureA.body in [self.env.body,self.env.head,self.env.tail]
                or contact.fixtureB.body in [self.env.body,self.env.head,self.env.tail]
        ):
            self.env.game_over = True



class envGym(gym.Env):
    def __init__(self,render='human'):
        self.render_mode=render
        #For making gif.
        # Window Title
        if self.render_mode != "speed":
            pygame.display.set_caption("environment")
            self.screen = pygame.display.set_mode((width, height))
        # Initialize display window.
        self.clock = pygame.time.Clock()
        # Control game frame rate to ensure simulation running speed is synchronized with real time.
        self.world = b2World(gravity=(0, 10), doSleep=True)
        # Gravity downwards, with a gravitational acceleration of 10
        # Inactive objects enter a sleep state,
        # Reducing the consumption of computing resources.
        self.objs = []
        # Store body of agent.
        self.body: Optional[Box2D.b2Body] = None
        # Main body.
        self.head: Optional[Box2D.b2Body] = None
        self.tail: Optional[Box2D.b2Body] = None
        self.joints = []
        # Store joints
        self.grounds = []
        # Store ground.
        self.pre_shaping = None
        # The shape of the previous state.
        self.game_over = False
        # Game End Flag

        # State space,
        # including subject horizontal and vertical velocities,
        # four joint angles, and four joint velocities
        low = np.array(
            [
                -5.0,
                -5.0,
                -math.pi,
                -math.pi,
                -math.pi,
                -math.pi,
                -5.0,
                -5.0,
                -5.0,
                -5.0,

            ]
        ).astype(np.float32)

        high = np.array(
            [
                5.0,
                5.0,
                math.pi,
                math.pi,
                math.pi,
                math.pi,
                5.0,
                5.0,
                5.0,
                5.0,
            ]
        ).astype(np.float32)
        # [
        # Horizontal velocity.
        # Vertical velocity of the main body.
        # 4 joints degree.
        # 4 joints velocities.]
        self.observation_space = spaces.Box(low, high)
        # The range of values for the velocity of the four joints in the action space is [-1, 1].
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
    def _destroy(self):
        if self.render_mode != "speed":
            pygame.display.quit()
        # Destroy current objects and ground during environment reset to clean up the scene.
        self.world.contactListener = None
        # Close listener.
        for ground in self.grounds:
            self.world.DestroyBody(ground)
            # Destroy static object (ground).
        for obj in self.objs:
            self.world.DestroyBody(obj)
            # Destroy body of agent.
        self.objs = []
        self.body = None
        self.head = None
        self.tail = None
        self.joints = []
        self.grounds = []

    def step(self, action: np.ndarray):
        thigh_speed = 18
        shank_speed = 14
        # Setting joint speed is 6.
        self.joints[0].motorSpeed = float(thigh_speed * np.sign(action[0]))
        self.joints[0].maxMotorTorque = float(120 * np.clip(np.abs(action[0]), 0, 1))
        self.joints[1].motorSpeed = float(shank_speed * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(120 * np.clip(np.abs(action[1]), 0, 1))
        self.joints[2].motorSpeed = float(thigh_speed * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(120 * np.clip(np.abs(action[2]), 0, 1))
        self.joints[3].motorSpeed = float(shank_speed * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(120 * np.clip(np.abs(action[3]), 0, 1))
        self.world.Step(1.0 / 60, 6 * 30, 2 * 30)
        # Update the simulation world with a step size of 1/60.
        pos = self.body.position
        # Obtain the location of the subject.
        # The state includes the speed of the subject, four joint angles, and velocity.
        state = [
            self.body.linearVelocity.x,
            self.body.angularVelocity,
            self.joints[0].angle,
            self.joints[1].angle,
            self.joints[2].angle,
            self.joints[3].angle,
            self.joints[0].speed / thigh_speed,
            self.joints[1].speed / shank_speed,
            self.joints[2].speed / thigh_speed,
            self.joints[3].speed / shank_speed,
        ]

        shaping = 150 * pos[0] / PPM
        # Location based rewards.
        reward = 0
        if self.pre_shaping is not None:
            reward = shaping - self.pre_shaping
            # Calculate reward.
        self.pre_shaping = shaping
        # Update the previous status.

        for a in action:
            reward -= 0.0001 * 50 * np.clip(np.abs(a), 0, 1)
            # Impose punishment.
        terminated = False
        # Initialization termination flag.
        if self.game_over or pos[0] < 0:
            reward = -100
            # Negative rewards will be given when the game ends or the subject's position is less than 0.
            terminated = True
        if pos[0] > (140 * 0.5):
            terminated = True
            # Set a termination flag when the subject reaches the endpoint.
        return np.array(state, dtype=np.float32), reward, terminated, {}
        # return state, reward, flag, Termination flag, truncation flag, and additional information.

    def reset( self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        # Initialize the entire scene.
        self._destroy()
        # Destroy everything.
        if self.render_mode != "speed":
            pygame.display.set_caption("environment")
            self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None

        self.scroll = 0.0
        start_ground_length = 150
        # The length of ground is 150 * 0.5 = 75 meter.
        y = init_height
        # height / 4 / PPM * 3
        relief = 0.0
        # The height fluctuations of the ground.
        ground_x = []
        # Store x axis
        ground_y = []
        # Store y axis
        for i in range(start_ground_length):
            ground_x.append(i * step_size)
            # Generate x-axis and y-axis coordinates for each ground segment.
            # Keep the first 10 ground segments flat
            relief = relief * math.sin(init_height - y) * 0.7
            # Randomly varying height, simulating undulating ground.
            if i > platform_length:
                # Randomly varying height, simulating undulating ground.
                relief += random.randint(-1, 1) * 2 / PPM
            y += relief
            ground_y.append(y)
        #  Each ground segment is defined by the coordinates of two points that define the edge.
        for i in range(start_ground_length - 1):
            g = self.world.CreateStaticBody(fixtures=b2FixtureDef(
                shape=b2EdgeShape(vertices=[(ground_x[i], ground_y[i]), (ground_x[i + 1], ground_y[i + 1])]),  # 边缘顶点
                friction=friction,
                # Friction
                categoryBits=0x01,
                # Category
            ))
            self.grounds.append(g)

        init_x = step_size * platform_length / 2
        # The center of the platform.
        init_y = init_height - 4 * leg_h - 2 * body_h
        # The height of agent should ensure that it is suspended above the ground.
        body = self.world.CreateDynamicBody(
            # Generate dynamic objects,
            # the body of agent is rectangle
            # indicating that they are subject to gravity and other physical influences.
            position=(init_x, init_y),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(body_w, body_h)),
                density=3.0,
                friction=0.1,
                categoryBits=0x10,
                maskBits=0x01,
                restitution=0.0,
            )
        )
        legs = []
        joints = []
        head = self.world.CreateDynamicBody(
            # The head of agent is a boll。
            position=(init_x + (body_w / 2), init_y),
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=0.5),
                density=0.1,
                friction=0.1,
                categoryBits=0x10,
                maskBits=0x01,
                restitution=0.0,
            )
        )
        # The body and head of agent are welded together to form a whole.
        # This means that there will be no relative movement between the body and the head.
        jd = b2WeldJointDef(
            bodyA=body,
            bodyB=head,
            localAnchorA=(body_w + 1, - 0.5),
            localAnchorB=(0.5, -0.25),
        )
        hb_jd = self.world.CreateJoint(jd)
        # The tail of agent is also a rectangle.
        tail = self.world.CreateDynamicBody(
            position=(init_x - (body_w / 2), init_y),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(body_w / 2, body_h / 2)),
                density=0.5,
                friction=0.1,
                categoryBits=0x10,
                maskBits=0x01,
                restitution=0.0,
            )
        )
        # The tail can rotate around body, it also a part of agent.
        jd = b2RevoluteJointDef(
            bodyA=body,
            bodyB=tail,
            localAnchorA=(- body_w - body_w / 2, body_h / 2),
            localAnchorB=(0, 0),
        )
        bt_jd = self.world.CreateJoint(jd)
        # Adding color for agent
        tail.color = (123, 104, 238)
        head.color = (123, 104, 238)
        body.color = (123, 104, 238)

        # Apply a random horizontal force to the body of the robotic dog for test,
        # causing it to have a random initial motion state at the beginning of the simulation
        body.ApplyForceToCenter(
            (random.randint(-1, 1), 0), True
        )
        # Define the physical shape and character of legs.
        leg_fd = b2FixtureDef(
            shape=b2PolygonShape(box=(leg_w, leg_h)),
            density=1.0,
            friction=friction,
            categoryBits=0x10,
            maskBits=0x01,
        )
        # Generate two thighs for the robotic dog.
        for i in range(-1, 2, 2):
            # -1 is left, 1 is right.
            bleg = self.world.CreateDynamicBody(
                position=(init_x, init_y + leg_h + body_h),
                angle=i * -0.05,
                fixtures=leg_fd
            )
            bleg.color = (106, 90, 205)
            # Connected to the body through rotating joints.
            jd = b2RevoluteJointDef(
                bodyA=body,
                bodyB=bleg,
                localAnchorA=(-i * body_w, body_h),
                localAnchorB=(0, - leg_h),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=80,
                # motorSpeed is transform joint.
                motorSpeed=1,
                # The angle limit of the joint is -0.8 to 1.1.
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            legs.append(bleg)
            joints.append(self.world.CreateJoint(jd))

            # The joint connecting the calf and calf to the thigh.
            sleg = self.world.CreateDynamicBody(
                position=(init_x + -(i * body_w), init_y + 2 * leg_h + body_h + leg_h),
                angle=i * 0.05,
                fixtures=leg_fd
            )
            sleg.color = (132, 112, 255)
            jd = b2RevoluteJointDef(
                bodyA=bleg,
                bodyB=sleg,
                localAnchorA=(0, leg_h),
                localAnchorB=(0, -leg_h),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=80,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            sleg.ground_contact = False
            legs.append(sleg)
            joints.append(self.world.CreateJoint(jd))
        self.body = body
        self.head = head
        self.tail = tail
        self.joints = joints + [hb_jd, bt_jd]
        self.objs = [body, head, tail] + legs
        # Store body, head, tail and legs in list for render.
        # return [body] + legs
        return self.step(np.array([0, 0, 0, 0]))[0]


    def render(self, mode="human"):
        if mode == "speed":
            return
        # Obtain the offset of the camera to keep the subject in the center of the screen at all times.
        # camera_offset_x = width / 2 - self.body.position[0] * PPM
        camera_offset_x = - (self.body.position[0] -4) * PPM
        camera_offset_y = 0

        self.screen.fill((66, 106, 179))

        #
        for ground in self.grounds:
            v = ground.fixtures[0].shape.vertices
            # Obtain the edge vertices of terrain objects.
            sp = [v[0], v[1], (v[1][0], height / PPM), (v[0][0], height / PPM)]
            # The first two points are the two vertices of the terrain,
            # and the last two points pull the y-coordinate to the bottom to form a rectangular ground.
            sc_sp = [(x * PPM + camera_offset_x, y * PPM + camera_offset_y) for (x, y) in sp]
            # Convert meters in the physical world to pixels.
            pygame.draw.polygon(self.screen, (255, 236, 139), sc_sp)
            # Convert meters in the physical world to pixels.

        for i in self.objs:
            for f in i.fixtures:
                trans = f.body.transform
                # The shape of an object.
                if type(f.shape) is b2CircleShape:
                    center = trans * f.shape.pos * PPM
                    center = (center[0] + camera_offset_x, center[1] + camera_offset_y)
                    pygame.draw.circle(self.screen, color=i.color, center=center, radius=f.shape.radius * PPM)
                else:
                    p = [((trans * v * PPM)[0] + camera_offset_x, (trans * v * PPM)[1] + camera_offset_y) for v in f.shape.vertices]
                    pygame.draw.polygon(self.screen, color=i.color, points=p)

        self.clock.tick(60)
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            pygame.display.flip()
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))[:, -width:]


    # def render(self, mode="human"):
    #     self.screen.fill((66, 106, 179))
    #     # Simulate the Sky
    #     for ground in self.grounds:
    #         v = ground.fixtures[0].shape.vertices
    #         # Obtain the edge vertices of terrain objects.
    #         sp = [v[0], v[1], (v[1][0], height / PPM), (v[0][0], height / PPM)]
    #         # The first two points are the two vertices of the terrain,
    #         # and the last two points pull the y-coordinate to the bottom to form a rectangular ground.
    #         sc_sp = [(x * PPM, y * PPM) for (x, y) in sp]
    #         # Convert meters in the physical world to pixels.
    #         pygame.draw.polygon(self.screen, (255, 236, 139), sc_sp)
    #         # The color of ground is brown.
    #     # Drawing body.
    #     for i in self.objs:
    #         for f in i.fixtures:
    #             # The shape of an object.
    #             trans = f.body.transform
    #             # Convert the local coordinates of an object to world coordinates.
    #             if type(f.shape) is b2CircleShape:
    #                 pygame.draw.circle(
    #                     self.screen,
    #                     color=i.color,
    #                     center=trans * f.shape.pos * PPM,
    #                     radius=f.shape.radius * PPM
    #                 )
    #             else:
    #                 p = [trans * v * PPM for v in f.shape.vertices]
    #                 pygame.draw.polygon(self.screen, color=i.color, points=p)
    #         pass
    #     self.clock.tick(60)
    #     # The rendering frame rate is 60 frames per second.
    #     # self.world.Step(1.0 / 60, 6, 2)
    #     # pygame.display.flip()
    #     # pass
    #     if mode == "human":
    #         pygame.display.flip()
    #     elif mode == "rgb_array":
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
    #         )[:, -width:]
    #     pass