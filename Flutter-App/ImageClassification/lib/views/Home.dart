import 'dart:math';

import 'package:ImageClassification/views/MathEquation.dart';
import 'package:ImageClassification/views/TextAndSpeech.dart';
import 'package:ImageClassification/widgets/wave.dart';
import 'package:flutter/material.dart';
import './SudokuSolver.dart';
import 'Barcode.dart';

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
    functions = ['SudokuSolver', 'MathEquation', 'Barcode', 'TextAndSpeech'];
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Processing'),
      ),
      body: Stack(
        children: [
          functionList(),
          onBottom(AnimatedWave(
            height: 180,
            speed: 1.0,
          )),
          onBottom(AnimatedWave(
            height: 120,
            speed: 0.9,
            offset: pi,
          )),
          onBottom(AnimatedWave(
            height: 220,
            speed: 1.2,
            offset: pi / 2,
          )),
        ],
      ),
    );
  }

  onBottom(Widget child) => Positioned.fill(
        child: Align(
          alignment: Alignment.bottomCenter,
          child: child,
        ),
      );
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
          } else if (this.function == 'MathEquation') {
            return MathEquation();
          } else if (this.function == 'Barcode') {
            return Barcode();
          } else if (this.function == 'TextAndSpeech') {
            return TextAndSpeech();
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
