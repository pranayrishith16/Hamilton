from pdf2image import convert_from_path
import pytesseract
from typing import List
from .interfaces import RawPage, Parser

class OCRPDFParser(Parser):
    def __init__(self,dpi:int=300,lang:str="eng"):
        self.dpi = dpi
        self.lang = lang
    
    def parse(self,path:str) -> List[RawPage]:
        images = convert_from_path(path,self.dpi)
        pages:List[RawPage] = []

        for idx,img in enumerate(images,start=1):
            text = pytesseract.image_to_string(img,lang=self.lang)
            pages.append(RawPage(page_number=idx,text=text.split(),metadata={'source':path,'ocr':True,'dpi':self.dpi}))
        return pages