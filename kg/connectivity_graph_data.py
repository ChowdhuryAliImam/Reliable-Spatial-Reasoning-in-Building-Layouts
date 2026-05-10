graph_1 = {"front_porch":["living_room"],
        "living_room":["front_porch","dining_room", "hall"],
        "dining_room":["living_room","kitchen"],
        "kitchen":["dining_room","service_porch"],
        "service_porch":["kitchen"],
        "hall":["living_room","bath_room","bed_room_1", "bed_room_2"],
        "bath_room":["hall"],
        "bed_room_1":["hall"], 
        "bed_room_2":["hall"]}


graph_2 ={"living_room":["hall","dining_room"],
          "hall":["living_room","hall_2","bath_room","bed_room_1", "bed_room_2","stair"],
         "dining_room":["living_room","hall_2"],
          "hall_2":["hall","dining_room", "kitchen"],
          "kitchen":["hall_2"],
          "bath_room":["hall"],
          "bed_room_1":["hall"],
          "bed_room_2":["hall"],
          "stair":["hall"]}



graph_3={"front_porch":["living_room"],
        "living_room":["front_porch","dining_room","bed_room_1", "hall"],
        "dining_room":["living_room","kitchen"],
        "kitchen":["dining_room","service_porch", "hall"],
        "service_porch":["kitchen"],
        "hall":["living_room","stair","bath_room","bed_room_1", "bed_room_2", "kitchen"],
        "stair":["hall"],
        "bath_room":["hall"],
        "bed_room_1":["living_room","hall"],
        "bed_room_2":["hall"]
}

graph_4={"front_porch":["living_room"],
         "living_room":["front_porch","dining_room","sun_room"],
         "dining_room":["living_room","hall","kitchen"],
         "sun_room":["living_room","kitchen"],
         "hall":["dining_room","stair","bath_room","kitchen","bed_room_1", "bed_room_2"],
         "stair":["hall"],
         "bath_room":["hall"],
         "kitchen":["hall","dining_room","sun_room", "service_entry"],
         "service_entry":["kitchen"],
         "bed_room_1":["hall","sleeping_porch"],
         "bed_room_2":["hall","sleeping_porch"],
         "sleeping_porch":["bed_room_1","bed_room_2"]}


graph_5 = {"landing":["front_porch_1", "front_porch_2"],
              "front_porch_1":["landing","living_room_1"],
              "front_porch_2":["landing","living_room_2"],
              "landing":["front_porch_1", "front_porch_2", "stair"],
              "living_room_1":["front_porch_1","dining_room_1","stair"],
              "living_room_2":["front_porch_2","dining_room_2","stair"],
              "dining_room_1":["living_room_1","hall_1"],
              "dining_room_2":["living_room_2","hall_2"],
              "hall_1":["dining_room_1","bed_room_1", "bed_room_2","kitchen_1","bath_room_1"],
              "hall_2":["dining_room_2","bed_room_3", "bed_room_4","kitchen_2","bath_room_2"],
              "bed_room_1":["hall_1"],
              "bed_room_2":["hall_1"],
              "bed_room_3":["hall_2"],
              "bed_room_4":["hall_2"],
              "kitchen_1":["hall_1", "service_porch_1"],
              "kitchen_2":["hall_2","service_porch_2"],
              "service_porch_1":["kitchen_1"],
              "service_porch_2":["kitchen_2"],
              "stair":["living_room_1","living_room_2", "landing"],
              "bath_room_1":["hall_1"],
              "bath_room_2":["hall_2"]}

