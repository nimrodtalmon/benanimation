# Drawing kit for Ben

The animation bends Ben's drawings (warp puppets, `puppet/`): the computer opens the mouth,
closes the eyes, turns and tilts the head, and bends the arms. So each character needs only
a few pages. The app for uploading them: talmonlab.bgu.ac.il/ben/ (code in `ben-app/`).

## One character = these pages
1. Head, front view, with the neck. Eyes open, mouth closed.
2. Eyes closed (only the lid lines), same size as on the head.
3. Mouths, 4 on one page: closed, a little open, wide open, round "ooh".
4. Body, front view, shoulders to hips, no arms. The whole body on the page.
5. Upper arm, shoulder to elbow, straight down, round ends.
6. Forearm, elbow to wrist with the cuff, no hand.
7. Hand holding a stick (fist, back of the hand to us).
8. Hand holding a mic (fist, thumb on top).
9. Open hand, relaxed.
10. Leg and shoe (optional).

Plain white paper, closed black outlines, all parts at the same scale, photo from straight above.
His own mouths, closed eyes and hands replace the code stand-ins (they looked better every time).

## How the parts are used
- Head + body: one warp puppet (`puppet/warp.py`): landmarks for face oval, eyes, mouth, chin, neck.
- Upper arm + forearm: joined at the elbow and bent with 2-bone IK (`puppet/limb.py`); the round
  elbow end of the forearm becomes the crease of the bent arm. One drawing serves both arms
  (the second is mirrored).
- Instruments and props are drawn in code (`puppet/props.py`).
