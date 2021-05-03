from flask import Flask,request,jsonify
import cv2
import os
import json

from .Translator.translate import Translator
from .Sudoku_Solver.Grid_Detection.Optimised import SudokuSolver
from .Math_Equation_Solver.math_equation_solver import MathEquationSolver
from .Barcode_Product.barcode_to_product_details import BarcodeToProductDetails

app=Flask(__name__)

@app.route('/Sudoku',methods=['POST','GET'])
def Sudoku():

    upload=request.files['picture']

    if(upload.filename != ''):
        upload.save(upload.filename)

        img=cv2.imread(upload.filename)
        os.remove(upload.filename)

        solver=SudokuSolver(img)
        solvedSudoku=solver.solveSudoku()

        print(solvedSudoku)
        d={'data':solvedSudoku}

        return jsonify(d)

@app.route('/MathEquation',methods=['POST','GET'])
def MathEquation():

    upload=request.files['picture']

    if(upload.filename != ''):
        upload.save(upload.filename)

        img=cv2.imread(upload.filename)
        os.remove(upload.filename)

        solver=MathEquationSolver()
        equation=solver.solveEquation(img)

        print(equation)
        d={'data':equation}

        return jsonify(d)

@app.route('/Barcode',methods=['POST','GET'])
def Barcode():

    upload=request.files['picture']

    if(upload.filename != ''):
        upload.save(upload.filename)

        img=cv2.imread(upload.filename)
        os.remove(upload.filename)

        solver=BarcodeToProductDetails(img)
        details=solver.getProductInformation()

        print(details)
        d={'data':details}

        return jsonify(d)

@app.route('/Translate',methods=['POST'])
def Translate():
    # takes body as simple string
    a=request.get_data(as_text=True)

    # deserializing string to json
    js=json.loads(a)

    trans=Translator()
    data=trans.translate(str(js['from']),str(js['to']),str(js['text']))
    d={'data':data}

    return jsonify(d)


if __name__=="main":
    app.run(debug=True)

# app.run(port=8000)