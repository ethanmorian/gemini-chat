import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';

import '../../model/movie_model.dart';

class MainCarouselSlider extends StatelessWidget {
  final List<MovieModel> topratedMovie;

  const MainCarouselSlider({super.key, required this.topratedMovie});

  @override
  Widget build(BuildContext context) {
    return CarouselSlider(
      items: topratedMovie.map(
        (movie) {
          return GestureDetector(
            onTap: () {},
            child: Image.network(
              'https://image.tmdb.org/t/p/w500/${movie.backdropPath}',
              fit: BoxFit.cover,
              width: MediaQuery.of(context).size.width,
            ),
          );
        },
      ).toList(),
      options: CarouselOptions(
        height: 500,
        autoPlay: true,
        aspectRatio: 16 / 9,
        autoPlayCurve: Curves.fastOutSlowIn,
        autoPlayAnimationDuration: const Duration(seconds: 1),
        enableInfiniteScroll: true,
        pageSnapping: true,
        enlargeCenterPage: true,
        viewportFraction: 1.0,
      ),
    );
  }
}
