"""
Preset emergency incident scenarios for realistic live demonstrations.
Each preset contains multimodal data: visual description, emergency transcript,
GPS coordinates, environmental sensors, and expected triage priority.
"""

PRESET_SCENARIOS = {
    "fire_outbreak": {
        "id": "INC-2026-9041",
        "title": "Industrial Warehouse Fire & Chemical Hazmat",
        "category": "Structure Fire / Toxic Gas",
        "image_url": "https://images.unsplash.com/photo-1542385151-efd9000785a0?w=600&auto=format&fit=crop&q=80",
        "transcript": "MAYDAY! Warehouse sector 4 is on fire, thick black noxious smoke filling the facility! Multiple workers trapped near loading dock 2. Chemical storage containers are beginning to rupture! We need immediate emergency rescue and Hazmat units!",
        "gps": {"lat": 12.9716, "lng": 77.5946, "address": "Ind. Corridor Block B, Sector 4"},
        "nearest_facility": "City General Hospital & Trauma Center (2.4 km)",
        "iot": {
            "temperature_c": 115, # severe heat
            "smoke_ppm": 880,     # toxic density
            "gas_leak_ppm": 420,  # hazardous chemical
            "structural_vibration": 0.88,
            "weather": "High Winds (28 km/h), Dry",
            "time_context": "Night Shift (02:15 AM)"
        },
        "ground_truth": "P1 - Critical",
        "expected_class": 0
    },
    "highway_collision": {
        "id": "INC-2026-9042",
        "title": "Multi-Vehicle Pileup on Expressway",
        "category": "High-Speed Transit Accident",
        "image_url": "https://images.unsplash.com/photo-1566008885218-90abf9200ddb?w=600&auto=format&fit=crop&q=80",
        "transcript": "Severe 4-car pileup on the northbound express highway! Two vehicles severely crumpled with victims trapped inside and unresponsive. Fuel leaking across both lanes with fire hazard. Traffic completely blocked, send advanced life support now!",
        "gps": {"lat": 12.9352, "lng": 77.6245, "address": "Outer Ring Expressway KM 18"},
        "nearest_facility": "Apollo Speciality Trauma Hospital (3.1 km)",
        "iot": {
            "temperature_c": 48,
            "smoke_ppm": 320,
            "gas_leak_ppm": 650, # High fuel vapor
            "structural_vibration": 0.92,
            "weather": "Heavy Rain, Poor Visibility (300m)",
            "time_context": "Peak Rush Hour (08:45 AM)"
        },
        "ground_truth": "P1 - Critical",
        "expected_class": 0
    },
    "urban_flood": {
        "id": "INC-2026-9043",
        "title": "Severe Flash Flood & Inundation",
        "category": "Natural Disaster / Hydrological",
        "image_url": "https://images.unsplash.com/photo-1547683905-f686c993aae5?w=600&auto=format&fit=crop&q=80",
        "transcript": "River embankment breached! Water level rising rapidly over 4 feet in the residential subway market. Ground floor shops submerged. Approximately 15 elderly residents stranded on second floor roofs, power lines submerged in water.",
        "gps": {"lat": 12.9822, "lng": 77.5812, "address": "Lowland Market Street, Riverbank Ward"},
        "nearest_facility": "Metropolitan Disaster Relief Center (4.5 km)",
        "iot": {
            "temperature_c": 22,
            "smoke_ppm": 20,
            "gas_leak_ppm": 10,
            "structural_vibration": 0.45,
            "weather": "Torrential Downpour (120 mm/hr)",
            "time_context": "Late Evening (07:10 PM)"
        },
        "ground_truth": "P2 - High",
        "expected_class": 1
    },
    "building_collapse": {
        "id": "INC-2026-9044",
        "title": "Commercial Annex Partial Collapse",
        "category": "Structural Failure",
        "image_url": "https://images.unsplash.com/photo-1584483766114-2cea6facdf57?w=600&auto=format&fit=crop&q=80",
        "transcript": "The exterior scaffolding and concrete balcony of the commercial building under renovation just collapsed onto the sidewalk! Rubble blocking the access lane, at least 3 pedestrians injured by falling bricks, bleeding with fractures.",
        "gps": {"lat": 12.9123, "lng": 77.6011, "address": "14 Commercial Road, Central District"},
        "nearest_facility": "St. John Emergency Medical Center (1.8 km)",
        "iot": {
            "temperature_c": 29,
            "smoke_ppm": 140, # Dust and pulverized debris
            "gas_leak_ppm": 35,
            "structural_vibration": 0.78,
            "weather": "Clear, Normal Winds",
            "time_context": "Afternoon (03:30 PM)"
        },
        "ground_truth": "P2 - High",
        "expected_class": 1
    },
    "kitchen_fire": {
        "id": "INC-2026-9045",
        "title": "Apartment Kitchen Stove Flare-Up",
        "category": "Localized Domestic Fire",
        "image_url": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=80",
        "transcript": "Cooking oil caught fire on the 3rd floor apartment kitchen. The stove is extinguished with fire extinguisher but heavy smoke is in the hallway. One resident inhaled smoke, coughing but fully conscious and ambulatory.",
        "gps": {"lat": 12.9554, "lng": 77.5621, "address": "Palm Grove Apartments, Flat 302"},
        "nearest_facility": "District Community Hospital (1.2 km)",
        "iot": {
            "temperature_c": 38,
            "smoke_ppm": 210,
            "gas_leak_ppm": 40,
            "structural_vibration": 0.12,
            "weather": "Clear, 28°C",
            "time_context": "Dinner Time (08:15 PM)"
        },
        "ground_truth": "P3 - Moderate",
        "expected_class": 2
    },
    "minor_fender_bender": {
        "id": "INC-2026-9046",
        "title": "Minor Low-Speed Parking Lot Scratch",
        "category": "Property Damage / Non-Emergency",
        "image_url": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=600&auto=format&fit=crop&q=80",
        "transcript": "Two cars grazed bumpers while backing out of the grocery parking spot. Both drivers are outside exchanging insurance. No injuries, airbags did not deploy, no fluids leaking. Requesting traffic constable for accident report.",
        "gps": {"lat": 12.9211, "lng": 77.6189, "address": "Supermarket East Parking, Zone C"},
        "nearest_facility": "Urban Health Clinic (0.8 km)",
        "iot": {
            "temperature_c": 27,
            "smoke_ppm": 12,
            "gas_leak_ppm": 0,
            "structural_vibration": 0.05,
            "weather": "Sunny, Calm",
            "time_context": "Midday (11:20 AM)"
        },
        "ground_truth": "P4 - Low",
        "expected_class": 3
    }
}
