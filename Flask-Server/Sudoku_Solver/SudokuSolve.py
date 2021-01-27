
class SudokuSolve:
    def __init__(self):
        pass
    def __find_empty_location(self,arr, l):
        for row in range(9):
            for col in range(9):
                if (arr[row][col] == 0):
                    l[0] = row
                    l[1] = col
                    return True
        return False


    def __used_in_row(self,arr, row, num):
        for i in range(9):
            if (arr[row][i] == num):
                return True
        return False

    def __used_in_col(self,arr, col, num):
        for i in range(9):
            if (arr[i][col] == num):
                return True
        return False

    def __used_in_box(self,arr, row, col, num):
        for i in range(3):
            for j in range(3):
                if (arr[i + row][j + col] == num):
                    return True
        return False

    def __check_location_is_safe(self,arr, row, col, num):
        return not self.__used_in_row(arr, row, num) and not self.__used_in_col(arr, col, num) and not self.__used_in_box(arr, row - row % 3,col - col % 3, num)

    def __solve_sudoku(self,arr):
        l = [0, 0]

        if (not self.__find_empty_location(arr, l)):
            return True

        row = l[0]
        col = l[1]

        for num in range(1, 10):
            if (self.__check_location_is_safe(arr,row, col, num)):
                arr[row][col] = num
                if (self.__solve_sudoku(arr)):
                    return True
                arr[row][col] = 0
        return False
    def solver(self,arr):
        if(self.__solve_sudoku(arr)):
            return arr
        else:
            return [[-1]*9]*9

if __name__ == "__main__":

    # assigning values to the grid
    grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
            [5, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 7, 0, 0, 0, 0, 3, 1],
            [0, 0, 3, 0, 1, 0, 0, 8, 0],
            [9, 0, 0, 8, 6, 3, 0, 0, 5],
            [0, 5, 0, 0, 9, 0, 6, 0, 0],
            [1, 3, 0, 0, 0, 0, 2, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 4],
            [0, 0, 5, 2, 0, 6, 3, 0, 0]]
    solver=SudokuSolve()
    solver.solve_sudoku(grid)
