import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

class LoadingScreen extends StatelessWidget {
  String image;
  String message;
  LoadingScreen({this.image, this.message});

  @override
  Widget build(BuildContext context) {
    final spinkit = SpinKitWave(
      itemBuilder: (BuildContext context, int index) {
        return DecoratedBox(
          decoration: BoxDecoration(
            color: Colors.white,
          ),
        );
      },
    );

    return Scaffold(
      backgroundColor: Colors.lightBlue[900],
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              this.message,
              style: TextStyle(
                fontFamily: "Capriola",
                color: Colors.white,
                fontSize: 20,
              ),
            ),
            SizedBox(height: 10.0),
            displayLoadingImage(image),
            SizedBox(height: 80.0),
            Align(
              alignment: FractionalOffset.bottomCenter,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Made with ',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 8.0,
                    ),
                  ),
                  Icon(
                    Icons.favorite,
                    color: Colors.red,
                    size: 10.0,
                  ),
                  Text(
                    ' in Flutter',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 8.0,
                    ),
                  ),
                ],
              ),
            ),
            // spinkit,
          ],
        ),
      ),
    );
  }

  Container displayLoadingImage(String imagePath) {
    return Container(
      height: 400,
      decoration: BoxDecoration(
        image: DecorationImage(
          image: AssetImage('assets/images/loading/' + imagePath),
          alignment: Alignment.center,
        ),
      ),
    );
  }
}
