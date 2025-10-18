Chassis

- Centroidal Frame:
  - Position: [0.0, 0.0, 0.0]
  - Orientation (Quaternion): [1.0, 0.0, 0.0, 0.0]
- Mass
- Moment of Inertia: [Ix, Iy, Iz]
- Products of Inertia: [Ixy, Ixz, Iyz]
- Void: true

- Driver Position:
  - Position: [x, y, z]
  - Orientation (Quaternion): [qw, qx, qy, qz]

don't know what rear connector is, maybe dont have

- Rear Connector Position: [x, y, z]

- Axles:

  - Front Double Wishbone

    - Location: [x, y, z]
    - Steering index: 0
    - Wishbone:

      - Camber angle: degrees
      - Toe angle: degrees
      - Spindle:

        - Mass
        - Inertia: [Ix, Iy, Iz]
        - COM: [x, y, z]
        - Radius: value
        - Width: value

      - Upright:

        - Mass: kg
        - COM: [x, y, z] m
        - Moments of Inertia: [Ix, Iy, Iz] kg\*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg\*m^2
        - Radius: m

      - Upper Control Arm:
        - Mass: kg
        - COM: [x, y, z] m
        - Moments of Inertia: [Ix, Iy, Iz] kg\*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg\*m^2
        - Radius: m
        - Location Chassis Front: [x, y, z] m
        - Location Chassis Back: [x, y, z] m
        - Location Upright: [x, y, z] m

      # two lower

      - Lower Control Arm:

        - Mass: kg
        - COM: [x, y, z] m
        - Moments of Inertia: [Ix, Iy, Iz] kg\*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg\*m^2
        - Radius: m
        - Location Chassis Front: [x, y, z] m
        - Location Chassis Back: [x, y, z] m
        - Location Upright: [x, y, z] m

      - Tierod:

        - Location Chassis: [x, y, z] m
        - Location Upright: [x, y, z] m

      - Spring:

        - Location Chassis: [x, y, z] m
        - Location Arm: [x, y, z] m
        - Free Length: m
        - Spring Coefficient: N/m

      - Shock:

        - Location Chassis: [x, y, z] m
        - Location Arm: [x, y, z] m
        - Damping Coefficient: N\*s/m

      - Axle:
        - Inertia: kg\*m^2

  - Whatever goes on in the front

- Wheels:
  - Mass: kg
  - Inertia: [Ix, Iy, Iz]

Rack Pinion

- Location: [x, y, z] m
- Orientation (Quaternion): [qw, qx, qy, qz]
- Steering Link:
  - Mass: kg
  - COM: [x, y, z] m
  - Moments of Inertia: [Ix, Iy, Iz] kg\*m^2
  - Radius: m
  - Length: m
- Pinion:
  - Radius: m
  - Maximum Angle: degrees'

Wheelbase: value m
Minimum turning radius: value m
Maximum steering angle: degrees

- Tires:
  if they have .tirs that is amazing and makes our life easy
  - Radius: m
  - Width: m
  - Mass: kg
  - Inertia: [Ix, Iy, Iz]
  - Contact Material:
    - Coefficent of friction: value
    - Restitution: value
    - Young's Modulus: value
    - Poisson Ratio: value
    - Normal Stiffness: value
    - Normal Damping: value
    - Tangential Stiffness: value
    - Tangential Damping: value

Power Train

- "Maximal Motor Speed RPM": RPM
- "Map Full Throttle": [RPM, Torque]
- "Map Zero Throttle": [RPM, Torque]

front suspension: double wishbone/control arms with Bell crank
rear: trailing arm 

what we have so far:
(from kayla kinematics file)

Chassis:
  Centroidal Frame:
    Position: [0.0, 0.0, 0.0]
    Orientation (Quaternion): [1.0, 0.0, 0.0, 0.0]
  Mass: TBD
  Moment of Inertia: [TBD, TBD, TBD]
  Products of Inertia: [TBD, TBD, TBD]
  Void: true

  Driver Position:
    Position: [TBD, TBD, TBD]
    Orientation (Quaternion): [TBD, TBD, TBD, TBD]

  Rear Connector Position: null  # not used / unknown

Axles:

  Front Double Wishbone (with Bellcrank):
    Location: [TBD, TBD, TBD]
    Steering index: 0
    Wishbone:
      Camber angle (deg): 1        # from IA[FL]
      Toe angle (deg): 0           # from Toe[FL]

      Spindle:
        Mass: TBD
        Inertia: [TBD, TBD, TBD]
        COM: [TBD, TBD, TBD]
        Radius: TBD
        Width: TBD

      Upright:
        Mass: TBD
        COM: [TBD, TBD, TBD]
        Moments of Inertia: [TBD, TBD, TBD]
        Products of Inertia: [TBD, TBD, TBD]
        Radius: TBD

      Upper Control Arm:
        Mass: TBD
        COM: [TBD, TBD, TBD]
        Moments of Inertia: [TBD, TBD, TBD]
        Products of Inertia: [TBD, TBD, TBD]
        Radius: TBD
        # Inboard points (to chassis): FLIP[Upper Fore], FLIP[Upper Aft]
        Location Chassis Front: [0.1143, 0.3175, 0.3485]
        Location Chassis Back:  [-0.1143, 0.3175, 0.3485]
        # Outboard (to upright): FLOP[Upper Fore] (representative)
        Location Upright: [-0.0191, 0.5970, 0.3858]

      Lower Control Arm:
        Mass: TBD
        COM: [TBD, TBD, TBD]
        Moments of Inertia: [TBD, TBD, TBD]
        Products of Inertia: [TBD, TBD, TBD]
        Radius: TBD
        # Inboard points (to chassis): FLIP[Lower Fore], FLIP[Lower Aft]
        Location Chassis Front: [0.1143, 0.2540, 0.1453]
        Location Chassis Back:  [-0.1143, 0.2540, 0.1453]
        # Outboard (to upright): FLOP[Lower Fore] (representative)
        Location Upright: [0.0191, 0.6264, 0.1463]

      Tierod:
        # Chassis-side: FLIP[Tie Rod]; Upright-side: FLOP[Tie Rod]
        Location Chassis: [-0.0508, 0.2669, 0.1866]
        Location Upright: [-0.0652, 0.5852, 0.1962]

      Spring:
        # Using bellcrank shock points to infer spring endpoints
        Location Chassis: [0.0191, 0.2369, 0.5720]   # FLBC[Shock Inboard]
        Location Arm:     [0.0191, 0.3494, 0.4274]   # FLBC[Shock Outboard]
        Free Length: TBD
        Spring Coefficient: TBD

      Shock:
        Location Chassis: [0.0191, 0.2369, 0.5720]   # same as above
        Location Arm:     [0.0191, 0.3494, 0.4274]
        Damping Coefficient: TBD

      Push/Pull Rod:
        Mounts to Upper Wishbone: False               # FL_Up

      Bellcrank:
        Pivot: [0.0, 0.3175, 0.3485]                 # FLBC[Pivot]
        Pivot Direction: [1.0, 0.0, 0.0]             # FLBC[Pivot Direction]
        Shock Outboard: [0.0191, 0.3494, 0.4274]
        Shock Inboard:  [0.0191, 0.2369, 0.5720]

      Axle:
        Inertia: TBD

    Wheel Contact Patch (Front Left): [0.0, 0.675, 0.0]   # FLCP

  Rear Trailing Arm:
    Location: [TBD, TBD, TBD]
    Steering index: 1
    Camber angle (deg): 0       # from IA[RL]
    Toe angle (deg): 0          # from Toe[RL]

    Links/Points (Left side):
      Inboard:
        Upper Fore: [-1.28834, 0.26000, 0.24500]
        Upper Aft:  [-1.50424, 0.26000, 0.24500]
        Lower Fore: [-1.28834, 0.25205, 0.08000]
        Lower Aft:  [-1.50424, 0.25205, 0.08000]
        Tie Rod:    [-1.3742004204848486, 0.2552116041818182, 0.1456182]
        Push/Pull Rod: [-1.47573900, 0.30957935, 0.11471471]
      Outboard (to upright/hub):
        Upper Fore: [-1.5501596, 0.57500, 0.2952369]
        Upper Aft:  [-1.5501596, 0.57500, 0.2952369]
        Lower Fore: [-1.5549860, 0.5775969, 0.10541]
        Lower Aft:  [-1.5549860, 0.5775969, 0.10541]
        Tie Rod:    [-1.45744993, 0.58648783, 0.1996660600320616]
        Push/Pull Rod: [-1.55040452, 0.51228235, 0.29667004]

    Push/Pull Rod mounts to Upper Wishbone: True   # RL_Up

    Bellcrank:
      Pivot: [-1.4732, 0.2794, 0.13335]
      Pivot Direction: [1.0, 0.0, 0.0]
      Shock Outboard: [-1.47840082, 0.25837024, 0.2]
      Shock Inboard:  [-1.4351, 0.07899653, 0.1397]

    Wheel Contact Patch (Rear Left): [-1.5494, 0.6096, 0.0]

Wheels:
  Mass: TBD
  Inertia: [TBD, TBD, TBD]

Rack Pinion:
  Location: [TBD, TBD, TBD]
  Orientation (Quaternion): [TBD, TBD, TBD, TBD]
  Steering Link:
    Mass: TBD
    COM: [TBD, TBD, TBD]
    Moments of Inertia: [TBD, TBD, TBD]
    Radius: TBD
    Length: TBD
  Pinion:
    Radius: TBD
    Maximum Angle: TBD

Wheelbase: TBD
Minimum turning radius: TBD
Maximum steering angle: TBD

Tires:
  Radius: TBD
  Width: TBD
  Mass: TBD
  Inertia: [TBD, TBD, TBD]
  Contact Material:
    Coefficent of friction: TBD
    Restitution: TBD
    Young's Modulus: TBD
    Poisson Ratio: TBD
    Normal Stiffness: TBD
    Normal Damping: TBD
    Tangential Stiffness: TBD
    Tangential Damping: TBD

Power Train:
  "Maximal Motor Speed RPM": TBD
  "Map Full Throttle": [[RPM, Torque], ...]  # TBD
  "Map Zero Throttle": [[RPM, Torque], ...]  # TBD
