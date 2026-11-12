import { staticFile } from "remotion";

/**
 * Distor, side view, drawn in code from Ben's front drawing.
 * Native drawing faces left; facing="right" mirrors it. The tears are on his
 * right cheek, so they show only when he faces right (his right side is visible).
 * Fills use texture patches cut from Ben's drawing (public/tex).
 * Coordinates: 1200 x 1600.
 */
export const DistorSide: React.FC<{
  uid: string;
  facing?: "left" | "right";
  blink?: number; // 0 open .. 1 closed
  mouth?: number; // 0 closed .. 1 open
  tilt?: number; // head tilt, degrees
}> = ({ uid, facing = "right", blink = 0, mouth = 0, tilt = 0 }) => {
  const id = (s: string) => `${uid}-${s}`;
  const url = (s: string) => `url(#${id(s)})`;
  const ink = { stroke: "#151313", strokeWidth: 7, strokeLinejoin: "round" as const, strokeLinecap: "round" as const };
  const line = { ...ink, fill: "none", strokeWidth: 5 };
  const mirrored = facing === "right";
  const eye = "M 424 578 Q 470 556 522 566 L 516 606 Q 470 614 432 604 Z";

  return (
    <svg viewBox="0 0 1200 1600" width="100%" height="100%" style={{ overflow: "visible" }}>
      <defs>
        <pattern id={id("skin")} patternUnits="userSpaceOnUse" width={252} height={510}>
          <image href={staticFile("tex/skin.png")} width={252} height={510} preserveAspectRatio="none" />
        </pattern>
        <pattern id={id("jacket")} patternUnits="userSpaceOnUse" width={300} height={380}>
          <image href={staticFile("tex/jacket.png")} width={300} height={380} preserveAspectRatio="none" />
        </pattern>
        <pattern id={id("hair")} patternUnits="userSpaceOnUse" width={1100} height={320} patternTransform="rotate(-8)">
          <image href={staticFile("tex/hair.png")} width={1100} height={320} preserveAspectRatio="none" />
        </pattern>
        {/* slightly uneven, hand-drawn outlines */}
        <filter id={id("wobble")} x="-5%" y="-5%" width="110%" height="110%">
          <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves={2} seed={5} />
          <feDisplacementMap in="SourceGraphic" scale={4} xChannelSelector="R" yChannelSelector="G" />
        </filter>
        <clipPath id={id("eye")}>
          <path d={eye} />
        </clipPath>
        <clipPath id={id("shirt")}>
          <polygon points="470,1110 545,1150 515,1600 440,1600" />
        </clipPath>
      </defs>

      <g transform={mirrored ? "translate(1200 0) scale(-1 1)" : undefined} filter={url("wobble")}>
        {/* head + neck, tilting around the neck */}
        <g transform={`rotate(${tilt} 640 1020)`}>
          <path
            {...ink}
            fill={url("skin")}
            d="M 430 300 L 420 470 Q 414 540 426 566 L 420 582 L 408 612 L 362 712
               Q 348 742 362 756 Q 378 766 398 760 Q 410 754 416 764 L 408 776
               Q 398 792 403 806 L 398 814 Q 402 828 406 836 Q 392 860 394 884 Q 398 924 424 936
               L 478 942 L 486 1000 L 498 1180 L 790 1180 L 776 905
               Q 870 700 868 500 Q 850 300 650 262 Q 500 250 430 300 Z"
          />
          {/* soft shade under the jaw */}
          <path fill="#c98a5e" opacity={0.28} d="M 430 940 L 605 905 L 655 800 L 780 880 L 790 1180 L 498 1180 L 486 1000 L 478 942 Z" />
          <path {...line} d="M 428 936 L 605 904 L 654 796" />
          <path {...line} strokeWidth={4} d="M 540 985 Q 552 1030 545 1070" />

          {/* face */}
          <path {...line} d="M 392 750 Q 404 740 414 754" />
          <path {...line} d="M 400 814 L 436 818" />
          {mouth > 0.02 && (
            <path {...ink} strokeWidth={4} fill="#3b1416" d={`M 400 812 L 436 817 L ${404 + 4 * mouth} ${818 + 22 * mouth} Z`} />
          )}
          <path {...line} d="M 484 645 Q 476 720 444 790" />

          {/* eye: heavy lid, purple shadow, no pupil */}
          <g transform="translate(470 590) scale(1.3) translate(-470 -590)">
          <path fill="#ebe9ef" d={eye} />
          <path fill="#cbc4d6" d="M 426 579 Q 470 559 520 568 L 518 586 Q 470 580 430 592 Z" />
          <path d="M 436 610 Q 475 620 515 607" fill="none" stroke="#5b2c7d" strokeWidth={9} strokeLinecap="round" />
          <path {...line} d="M 432 617 Q 475 629 520 612" />
          <g clipPath={url("eye")}>
            <rect x={400} y={540} width={140} height={20 + 70 * blink} fill={url("skin")} />
          </g>
          <path
            d="M 418 576 Q 468 548 528 563"
            transform={`translate(0 ${blink * 40})`}
            fill="none"
            stroke="#151313"
            strokeWidth={13}
            strokeLinecap="round"
          />
          </g>

          {mirrored && (
            <g stroke="#c4122f" strokeLinecap="round" fill="none" opacity={0.92}>
              <path d="M 440 620 C 446 660 449 705 452 770" strokeWidth={9} />
              <path d="M 458 624 C 463 650 463 670 465 694" strokeWidth={6} />
            </g>
          )}

          {/* hair: Ben's own hair texture, messy curled tips */}
          <path
            {...ink}
            fill={url("hair")}
            d="M 418 520 Q 441 498 440 466 Q 461 504 462 548 Q 492 523 494 484 Q 514 517 516 556 Q 546 532 548 494 Q 570 527 574 566 Q 602 544 604 508 Q 636 570 636 640 Q 612 707 646 770 Q 641 702 668 640 Q 689 611 682 576 Q 715 573 746 586 Q 756 630 796 652 Q 815 708 806 766 Q 833 743 834 708 Q 858 767 852 830 Q 887 777 872 716 Q 889 744 892 776 Q 925 698 890 620 Q 903 585 930 560 Q 931 524 905 500 Q 926 469 958 450 Q 942 419 908 410 Q 921 370 950 340 Q 924 317 890 320 Q 885 279 900 240 Q 865 238 840 262 Q 819 224 818 180 Q 783 195 770 232 Q 741 200 730 158 Q 696 181 690 222 Q 658 197 640 160 Q 609 186 608 226 Q 574 208 552 176 Q 528 209 536 250 Q 501 249 470 232 Q 456 261 468 290 Q 432 304 394 300 Q 399 332 426 350 Q 403 378 370 392 Q 390 412 418 410 Q 405 438 380 456 Q 388 475 408 482 Z"
          />

          {/* big ear, in front of the hair */}
          <path {...ink} fill={url("skin")} d="M 668 640 Q 690 590 742 600 Q 792 615 787 682 Q 783 742 746 772 Q 715 792 690 772 Q 674 760 679 744 Z" />
          <path {...line} d="M 700 652 Q 742 640 752 686 Q 750 722 726 738" />
          <path fill="#c98a5e" opacity={0.3} d="M 700 652 Q 742 640 752 686 Q 750 722 726 738 Q 712 700 700 652 Z" />
        </g>

        {/* shirt: black with red/grey flames */}
        <polygon {...ink} fill="#1b1b1d" points="470,1110 545,1150 515,1600 440,1600" />
        <g clipPath={url("shirt")}>
          <path fill="#5a5f68" d="M 440 1150 L 470 1185 L 480 1150 L 505 1200 L 515 1160 L 545 1215 L 545 1290 L 440 1290 Z" />
          <path fill="#cf1a2c" d="M 440 1200 L 462 1240 L 475 1205 L 492 1262 L 510 1220 L 520 1275 L 545 1250 L 545 1320 L 440 1320 Z" />
          <path fill="#1b1b1d" d="M 440 1300 L 470 1270 L 488 1310 L 505 1280 L 525 1315 L 545 1300 L 545 1340 L 440 1340 Z" />
        </g>

        {/* white jacket, stand-up collar with grey band */}
        <path {...ink} fill={url("jacket")} d="M 540 1150 L 830 1105 Q 960 1118 1022 1180 Q 1064 1250 1066 1600 L 515 1600 Z" />
        <path fill="#9b968c" opacity={0.18} d="M 860 1112 Q 960 1120 1022 1180 Q 1064 1250 1066 1600 L 900 1600 Q 930 1350 860 1112 Z" />
        <path {...line} d="M 880 1120 Q 935 1330 912 1600" />
        <path {...ink} fill={url("jacket")} d="M 495 1095 L 810 1040 L 835 1110 L 540 1158 Z" />
        <path fill="#a5a8b5" d="M 498 1098 L 808 1044 L 812 1062 L 503 1115 Z" />
        <path {...line} d="M 501 1114 L 812 1060" />
        <g {...ink} strokeWidth={4} fill="#e3b21f">
          <circle cx={572} cy={1212} r={12} />
          <circle cx={556} cy={1520} r={12} />
        </g>
      </g>
    </svg>
  );
};
