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
  - Back Double Wishbone
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
        - Moments of Inertia: [Ix, Iy, Iz] kg*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg*m^2
        - Radius: m

      - Upper Control Arm:
        - Mass: kg
        - COM: [x, y, z] m
        - Moments of Inertia: [Ix, Iy, Iz] kg*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg*m^2
        - Radius: m
        - Location Chassis Front: [x, y, z] m
        - Location Chassis Back: [x, y, z] m
        - Location Upright: [x, y, z] m

      - Lower Control Arm:
        - Mass: kg
        - COM: [x, y, z] m
        - Moments of Inertia: [Ix, Iy, Iz] kg*m^2
        - Products of Inertia: [Ixy, Ixz, Iyz] kg*m^2
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
        - Damping Coefficient: N*s/m

      - Axle:
        - Inertia: kg*m^2

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
  - Moments of Inertia: [Ix, Iy, Iz] kg*m^2
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