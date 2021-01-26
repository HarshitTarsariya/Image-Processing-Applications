import 'package:flutter/material.dart';
import './SudokuSolver.dart';

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  List<String> functions = [];
  Widget functionList() {
    return StreamBuilder(builder: (context, snapshot) {
      return ListView.builder(
        itemCount: this.functions.length,
        itemBuilder: (context, index) {
          return Function(this.functions[index]);
        },
      );
    });
  }

  @override
  void initState() {
    // TODO: implement initState
    functions = ['SudokuSolver', 'ImageToText'];
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Processing'),
      ),
      body: functionList(),
    );
  }
}

class Function extends StatelessWidget {
  final String function;
  Function(this.function);
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.push(context, MaterialPageRoute(builder: (context) {
          if (this.function == 'SudokuSolver') {
            return SudokuSolver();
          }
        }));
      },
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: Container(
            height: 120,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              gradient: LinearGradient(
                colors: [Colors.white, Colors.green],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey,
                  blurRadius: 12,
                  offset: Offset(0, 6),
                )
              ],
            ),
            child: Row(
              children: [
                Expanded(
                  flex: 2,
                  child: Image.asset(
                    'assets/${this.function}.png',
                    height: 80,
                    width: 80,
                  ),
                ),
                Expanded(
                  flex: 4,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        this.function,
                        style: TextStyle(
                          color: Colors.black87,
                          fontWeight: FontWeight.w500,
                          fontSize: 30,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
