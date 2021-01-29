from flask import Flask,request,jsonify
import cv2
import os
from .Sudoku_Solver.Grid_Detection.Optimised import SudokuSolver

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

if __name__=="main":
    app.run(debug=True)

# app.run(port=8000)