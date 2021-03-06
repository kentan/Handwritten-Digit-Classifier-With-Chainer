import os
from flask import Flask
from flask import request
import ast
import json
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
          vec,ans = predictor.predict(data);
          print(vec);
          rv = json.dumps({"vec":vec,"ans":str(ans)});
          return(rv);
      except Exception as ex:
          print(str(ex));
          return(str(ex)); 
      except Error as er:
          print(str(er));
          return(str(er));

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

