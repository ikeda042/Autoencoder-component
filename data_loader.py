from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import cv2
import numpy as np
import pickle
import os 


def load_database(output_dir:str, database_name: str) -> None:
    for dir in [output_dir, f'{output_dir}/ph', f'{output_dir}/fluo']:
        try:
            os.mkdir(dir)
        except:
            pass
    Base = declarative_base()
    class Cell(Base):
        __tablename__ = 'cells'
        id = Column(Integer, primary_key=True)
        cell_id = Column(String)
        label_experiment = Column(String)
        manual_label  = Column(Integer)
        perimeter = Column(FLOAT)
        area = Column(FLOAT)
        img_ph = Column(BLOB) 
        img_fluo1 = Column(BLOB)
        img_fluo2 = Column(BLOB)
        contour = Column(BLOB)
        center_x = Column(FLOAT)
        center_y = Column(FLOAT)
    engine = create_engine(f'sqlite:///{database_name}', echo=True)
    Session = sessionmaker(bind=engine)
    n = 0
    with Session() as session:
        cells = session.query(Cell).all()
        for cell in cells:
            if cell.manual_label == 1:
                image_ph = cv2.imdecode(np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(f"{output_dir}/ph/{n}.png", image_ph)
                n += 1
                image_fluo1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                cell_contour = [list(i[0]) for i in pickle.loads(cell.contour)]
                image_size = image_fluo1.shape
                mask = np.zeros((image_size[0],image_size[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [pickle.loads(cell.contour)], 1)
                output_image = cv2.bitwise_and(image_fluo1, image_fluo1, mask=mask)
                #輝度を0-255に正規化
                output_image = output_image - np.min(output_image)
                output_image = output_image / np.max(output_image) * 255
                output_image = output_image.astype(np.uint8)
                output_image_color = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
                output_image_color[:, :, 0] = 0
                output_image_color[:, :, 2] = 0
                cv2.imwrite(f"{output_dir}/fluo/{n}.png", output_image_color)
                
    