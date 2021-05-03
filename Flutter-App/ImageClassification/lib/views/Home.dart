import 'package:flutter/material.dart';
import '../constants.dart';
import '../style/style.dart';
import '../views/Help.dart';
import '../style/myClipper.dart';
import '../style/screenTitle.dart';
import '../views/TranslateHome.dart';
import 'dart:math';

import './MathEquation.dart';
import 'TextAndSpeech.dart';
import '../style/wave.dart';
import './SudokuSolver.dart';
import './Barcode.dart';

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  List<String> functionTitle = [];
  List<String> functionImage = [];
  List<Card> functionCard = [];

  Widget functionList() {
    return Expanded(
      child: GridView.builder(
        itemCount: this.functionTitle.length,
        itemBuilder: (BuildContext context, int index) {
          return makeCardTileForGrid(
              this.functionTitle[index], this.functionImage[index]);
        },
        primary: false,
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          mainAxisSpacing: 5,
          crossAxisSpacing: 5,
        ),
      ),
    );
  }

  @override
  void initState() {
    functionTitle = [
      SUDOKU_SOLVER,
      MATH_EQUATION_SOLVER,
      BARCODE_TO_PRODUCT_DETAILS,
      TEXT_AND_SPEECH,
      TRANSLATOR,
      HELP,
    ];
    functionImage = [
      'SudokuSolver.png',
      'MathEquation.png',
      'Barcode.png',
      'TextAndSpeech.png',
      'Translator.png',
      'Help.png',
    ];
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.lightBlue[900],
      body: Stack(
        children: [
          ClipPath(
            clipper: MyClipper(),
            child: Container(
              height: 260,
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: AssetImage("assets/images/static/ComputerVision.png"),
                  alignment: Alignment.topRight,
                  fit: BoxFit.fitWidth,
                ),
              ),
            ),
          ),
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
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(40.0),
              child: Column(
                children: [
                  SizedBox(
                    height: 180,
                  ),
                  Container(
                    height: 68,
                    // margin: EdgeInsets.only(bottom: 5.0),
                    child: ScreenTitle(text: "Image Processing App"),
                  ),
                  functionList(),
                ],
              ),
            ),
          ),
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

  Card makeCardTileForGrid(String title, String imageName) {
    var size = MediaQuery.of(context).size;
    return Card(
      color: Colors.lightBlue[100],
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10.0),
      ),
      elevation: 4.0,
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) {
                switch (title) {
                  case SUDOKU_SOLVER:
                    return SudokuSolver();
                  case MATH_EQUATION_SOLVER:
                    return MathEquation();
                  case BARCODE_TO_PRODUCT_DETAILS:
                    return Barcode();
                  case TRANSLATOR:
                    return TranslateHome();
                  case TEXT_AND_SPEECH:
                    return TextAndSpeech();
                  default:
                    return Help();
                }
              },
            ),
          );
        },
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              height: size.height * 0.11,
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: AssetImage('assets/images/static/' + imageName),
                  alignment: Alignment.topCenter,
                ),
              ),
            ),
            Text(
              title,
              style: getCardTextStyle(),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
