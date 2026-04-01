pip install -r requirements.txt

#terminal 1
python train_models.py
uvicorn main:app --reload --port 8000

#terminal 2
npm install express ejs axios body-parser 
npm start
