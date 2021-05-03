import 'dart:async';

import 'package:ImageClassification/style/loadingScreen.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import './views/Home.dart';

void main() async {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    waitingFunc();
  }

  final Future<FirebaseApp> _initialization = Firebase.initializeApp();
  bool waiter = false;
  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: _initialization,
      builder: (context, snapshot) {
        return MaterialApp(
          title: 'Image Processing App',
          debugShowCheckedModeBanner: false,
          theme: ThemeData(
            primarySwatch: Colors.lightBlue,
            fontFamily: "Capriola",
          ),
          home: (snapshot.connectionState == ConnectionState.done &&
                  waiter == true)
              ? Home()
              : LoadingScreen(image: 'ai.png', message: 'Image Processing App'),
        );
      },
    );
  }

  waitingFunc() async {
    await Timer(Duration(seconds: 3), () {
      setState(() {
        waiter = true;
      });
    });
  }
}
