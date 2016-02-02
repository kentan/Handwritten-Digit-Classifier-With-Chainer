import os
from flask import Flask
from flask import request
import ast
import predictor

app = Flask(__name__)

from flask import render_template

@app.route('/hw')
def index(name=None):
    return render_template('./index.html', name=name)

@app.route('/estimate', methods=['POST'])
def estimate():
    if request.method == 'POST':
      try:
          data = ast.literal_eval(request.data.decode("utf-8"));
          data = data["input"];
          rv = predictor.predict(data);
          return(str(rv));
      except Error as e:
          return(str(e));

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

