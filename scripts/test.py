import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import easyocr
reader = easyocr.Reader(['en'])