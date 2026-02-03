from ultralytics import SAM

model = SAM("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt")

# Single point prompt - segments object at specific location
results = model.predict(source="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png",
                        points=[900, 370], labels=[1])
results[0].show()

# Multiple points - segments single object with multiple point hints
results = model.predict(source="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png",
                        points=[[400, 370], [900, 370]], labels=[1, 1])
results[0].show()

# Box prompt - segments object within bounding box
results = model.predict(source="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png",
                        bboxes=[100, 150, 300, 400])
results[0].show()