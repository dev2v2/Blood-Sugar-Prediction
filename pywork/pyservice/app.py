#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 서버 관리용 fastapi 의존 라이브러리
import uvicorn
# pip install fastapi uvicorn

# fast api 라이브러리
# 비동기 처리가 가능한 파이선 웹 서버 라이브러리
from fastapi import FastAPI

# 머신러닝 모델 관리용 라이브러리
# 데이터 바이너리 저장용 라이브러리
# 데이터 타입을 그대로 보존
import pickle

# 데이터프레임 및 수 처리 라이브러리
import pandas as pd
import numpy as np

# 인터페이스 데이터 관리를 위한 라이브러리
from pydantic import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import uvicorn

from fastapi.middleware.cors import CORSMiddleware


# In[ ]:


origins = ["*"]


# In[ ]:


try:
    with open("mldtcore.dump", "rb") as fr:
        loadedModel = pickle.load(fr)
    # 파일이 성공적으로 로드되었을 때 할 작업을 여기에 추가합니다.
    print("모델을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except (pickle.UnpicklingError, EOFError):
    print("파일을 열거나 읽는 동안 문제가 발생했습니다.")
except Exception as e:
    print("알 수 없는 예외가 발생했습니다:", e)


# In[ ]:


# features = ['temperature', 'smm', 'pbf', 'tbw']
# label = ['fasting_glucose']
class InDataset(BaseModel):
    temperature: float
    smm: float
    pbf: float
    tbw: float


# In[ ]:


class InDataset2(BaseModel):
    fastingGlucose: float


# In[ ]:


try:
    app = FastAPI(title = "ML API")
except Exception as e:
    print(e)


# In[ ]:


# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# In[ ]:


try:
    @app.get("/")
    async def root():
        return {"message":"hello server is running"}
except Exception as e:
    print(e)


# In[ ]:


try:
    @app.post("/predict", status_code=200)
    async def predict_ml(x: InDataset):
        # 이중리스트 : 행
        testDf = pd.DataFrame([[x.temperature, x.smm, x.pbf, x.tbw]])
    
        try:
            # 예측결과가 여러 개일 때 view로 던질 때 리스트, 단일 값, 딕셔너리 등 지정해주기
            predictValue = int(loadedModel[0].predict(testDf)[0])
            interfaceResult = {"result": predictValue}
            return interfaceResult
        except Exception as e:
            print(e)
except Exception as e:
    print(e)


# In[ ]:


try:
    @app.post("/predict2", status_code=200)
    async def predict2_ml(x: InDataset2):
        testDf = pd.DataFrame([[x.fastingGlucose]])
    
        try:
            # 예측결과가 여러 개일 때 view로 던질 때 리스트, 단일 값, 딕셔너리 등 지정해주기
            predictValue = int(loadedModel[1].predict(testDf)[0])
            interfaceResult = {"result": predictValue}
            return interfaceResult
        except Exception as e:
            print(e)
except Exception as e:
    print(e)


# In[ ]:


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999, log_level="debug",proxy_headers=True, reload=True)
    #app = 파일이름
    #host 0000 : 모든 곳에서 받는다

