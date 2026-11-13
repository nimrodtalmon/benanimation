# Drawing kit for Ben

The animation bends Ben's drawings (warp puppets, `puppet/`): the computer opens the mouth,
closes the eyes, turns and tilts the head, and bends the arms. So each character needs only
a few pages. The app for uploading them: talmonlab.bgu.ac.il/ben/ (code in `ben-app/`).

## One character = these pages
1. Head, front view, with the neck. Eyes open, mouth closed.
2. Body, front view, shoulders to hips, with clothes, no arms. The whole body on the page
   (a drawing that runs off the paper edge ends in a straight cut).
3. Upper arm, shoulder to elbow, hanging straight down, round ends.
4. Forearm + hand, elbow to fingers, straight down, hand in a loose fist (it holds the mic,
   drumsticks or a pick), round end at the elbow.
5. Leg (optional), hip to shoe, only if the legs show.

Plain white paper, closed black outlines, all parts at the same scale, photo from straight above.

## How the parts are used
- Head + body: one warp puppet (`puppet/warp.py`): landmarks for face oval, eyes, mouth, chin, neck.
- Upper arm + forearm: joined at the elbow and bent with 2-bone IK (`puppet/limb.py`); the round
  elbow end of the forearm becomes the crease of the bent arm. One drawing serves both arms
  (the second is mirrored).
- Instruments and props are drawn in code (`puppet/props.py`).
