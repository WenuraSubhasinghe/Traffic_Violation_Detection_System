from numpy import ndarray
from torch import Tensor

from app.services.light_violation_detection.config import CLASSES, PALLETES

def current_color(n: int, recognizedColor) -> str:
    for key, values in recognizedColor.items():
        for data in values:
            if data[0] == n:
                return key
    return "unknown"  # default

def annotate (frame:ndarray, prediction:Tensor, recognizedColor={}, violationMode:bool=False, scale:float=.5, padding:int=6):

   from cv2 import rectangle, getTextSize, putText
   from torch import ceil, floor


   for n,row in enumerate(prediction):
      if violationMode:
         label = f'{row[5]*100:.2f}% {CLASSES[int(row[6])].title()} VIOLATES'
      else:
         if not recognizedColor:
            label = f'{row[4].to(int)}. {row[5]*100:.2f}% {CLASSES[int(row[6])].title()}'
            color = PALLETES[CLASSES[int(row[6])]]
         else:
            color_name = current_color(n, recognizedColor) or "unknown"
            label = f'{row[5]*100:.2f}% {color_name.upper()}'
            color = PALLETES.get(color_name, (255, 255, 255))  # default white


      x1 = int(ceil(row[0]))
      y1 = int(ceil(row[1]))
      x2 = int(floor(row[2]))
      y2 = int(floor(row[3]))
      rectangle(frame, (x1,y1), (x2,y2), color, padding//2)

      w_label, h_label = getTextSize(label, 0, scale, int(scale*2))[0]
      rectangle(frame, (x1, y1-h_label-2*padding), (x1+w_label+2*padding, y1), color, -1)
      
      putText(frame, label, (x1+padding, y1-padding), 0, scale, (0,0,0), int(scale*2), 16)
