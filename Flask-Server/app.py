from flask import Flask,request,jsonify
import cv2
import os
#importing ../Sudoku-Solver/Grid Detection/SudokuSolver.py

app=Flask(__name__)

@app.route('/Sudoku',methods=['POST','GET'])
def Sudoku():

    upload=request.files['picture']

    if(upload.filename != ''):
        upload.save(upload.filename)

        img=cv2.imread(upload.filename)
        solver=SudokuSolver(img)
        solvedSudoku=solver.solveSudoku()

        os.remove(upload.filename)

        arr=[[int(solvedSudoku[i][j]) for j in range(9)]for i in range(9)]
        print(arr)
        d={'data':arr}

        return jsonify(d)

if __name__=="main":
    app.run(debug=True)

# app.run(port=8000)