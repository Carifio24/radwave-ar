This repository stores everything needed for the (work-in-progress) animated Radcliffe Wave AR figure.

Basically, I've been iterating in stages on this, hence the various versioned files. Just for posterity:

* `sphere`: Here's where I worked out a grid-style triangulated sphere mesh that we can use. You can set the center, radius, and resolutions in theta and phi.

* `first_frame`: As the name suggests, this creates a glTF with the first frame of the animation. This was the result of my experimentation with adding multiple meshes.

* `animated_clusters`: This creates an animated glTF showing the moving cluster points. This can be our first basic implementation of animating (multiple) meshes.
