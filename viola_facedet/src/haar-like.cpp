#include <iostream>

#include "haar-like.h"

/**
 * Constructor
 * @param x X offset relative to subwindow
 * @param y Y offset relative to subwindow
 * @param w Constituent rectangle width
 * @param h Constituent rectangle height
 * @param type Feature type 1-5 
 */
Haarlike::Haarlike(int x, int y, int w, int h, int type) {
	this->x = x; 
	this->y = y; 
	this->w = w; 
	this->h = h; 
	this->type = type; 
}

/**
 * Constructor
 */
Haarlike::Haarlike() {

}

/**
 * Destructively scale a Haarlike relative to its base resolution
 * @param factor The factor by which to scale
 */
void Haarlike::scale(float factor) {
	this->w *= factor;
	this->h *= factor;
	this->x *= factor;
	this->y *= factor;
}