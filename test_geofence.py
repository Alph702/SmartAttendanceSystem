import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius (meters)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

# Test Data (School Location)
SCHOOL_LAT = 24.8607
SCHOOL_LON = 67.0011
SCHOOL_RADIUS = 150

def test_distance():
    # 1. Inside school (same point)
    d1 = haversine(24.8607, 67.0011, SCHOOL_LAT, SCHOOL_LON)
    print(f"Inside (0m): {d1:.2f}m - {'PASS' if d1 <= SCHOOL_RADIUS else 'FAIL'}")

    # 2. Inside school (100m away)
    # Approx 0.0009 degrees for 100m
    d2 = haversine(24.8607 + 0.0005, 67.0011, SCHOOL_LAT, SCHOOL_LON)
    print(f"Inside (~55m): {d2:.2f}m - {'PASS' if d2 <= SCHOOL_RADIUS else 'FAIL'}")

    # 3. Outside school (500m away)
    d3 = haversine(24.8607 + 0.0045, 67.0011, SCHOOL_LAT, SCHOOL_LON)
    print(f"Outside (~500m): {d3:.2f}m - {'PASS' if d3 > SCHOOL_RADIUS else 'FAIL'}")

if __name__ == "__main__":
    test_distance()
