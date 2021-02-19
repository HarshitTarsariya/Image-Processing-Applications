from flask import Flask,request,jsonify
import cv2
import os
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

        solver=MathEquationSolver(img)
        equation=solver.solveEquation()

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

if __name__=="main":
    app.run(debug=True)

# app.run(port=8000)