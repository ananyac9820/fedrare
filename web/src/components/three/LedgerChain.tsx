"use client";

import { Edges, Html, RoundedBox } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useRef, useState } from "react";
import * as THREE from "three";

const SPACING = 1.75;
const BLOCK: [number, number, number] = [1.15, 0.85, 0.85];
const LABELLED = 9; // only the most recent blocks get a DOM label

type BlockState = "ok" | "edited" | "broken";

// Light theme: paper-coloured blocks with teal edges; tampered blocks turn red.
const COLORS: Record<BlockState, { body: string; edge: string; link: string }> = {
  ok: { body: "#fffdf9", edge: "#2c6a64", link: "#2c6a64" },
  edited: { body: "#f6d5d5", edge: "#ae2330", link: "#ae2330" },
  broken: { body: "#fbecec", edge: "#d58c93", link: "#d58c93" },
};

function Block({ index, round, state, selected, animate, isNewest, onSelect }: {
  index: number;
  round: number;
  state: BlockState;
  selected: boolean;
  animate: boolean;
  isNewest: boolean;
  onSelect: (round: number) => void;
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  // New blocks grow in; everything else (and reduced motion) starts at full size.
  const born = useRef(animate && isNewest ? 0.001 : 1);

  useFrame((s, delta) => {
    if (!mesh.current) return;
    const lift = hovered || selected ? 1.06 : 1;
    if (animate) {
      born.current = Math.min(1, born.current + delta * 1.8);
      const eased = 1 - Math.pow(1 - born.current, 3);
      mesh.current.scale.setScalar(eased * lift);
      mesh.current.position.y = Math.sin(s.clock.elapsedTime * 0.7 + index * 0.6) * 0.04;
    } else {
      mesh.current.scale.setScalar(lift);
    }
  });

  const c = COLORS[state];
  return (
    <group position={[index * SPACING, 0, 0]}>
      <RoundedBox
        ref={mesh}
        args={BLOCK}
        radius={0.1}
        smoothness={4}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(round);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          setHovered(false);
          document.body.style.cursor = "";
        }}
      >
        <meshStandardMaterial
          color={selected && state === "ok" ? "#dcebe7" : c.body}
          roughness={0.55}
          metalness={0.02}
        />
        <Edges color={selected ? "#1d2929" : c.edge} threshold={20} />
      </RoundedBox>
      {index > 0 && (
        <mesh position={[-SPACING / 2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.03, 0.03, SPACING - BLOCK[0], 10]} />
          <meshBasicMaterial color={c.link} transparent opacity={state === "ok" ? 0.75 : 0.45} />
        </mesh>
      )}
    </group>
  );
}

function Chain({ rounds, visible, tamperedFrom, selected, onSelect, animate }: {
  rounds: number[];
  visible: number;
  tamperedFrom: number | null;
  selected: number | null;
  onSelect: (round: number) => void;
  animate: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  const shown = rounds.slice(0, visible);
  const targetX = -(visible - 1) * SPACING + 1.9;

  useFrame((_, delta) => {
    if (!group.current) return;
    if (animate) {
      group.current.position.x += (targetX - group.current.position.x) * Math.min(1, delta * 2.5);
    } else {
      group.current.position.x = targetX;
    }
  });

  const stateOf = (round: number): BlockState =>
    tamperedFrom === null || round < tamperedFrom ? "ok" : round === tamperedFrom ? "edited" : "broken";

  // Outer group tilts the chain around the world origin (where the newest block sits);
  // the inner group only slides. Tilting the sliding group would pivot around the first
  // block and swing the newest ones out of frame as the chain grows.
  return (
    <group rotation={[0.1, -0.5, 0]} position={[0.2, -0.2, 0]}>
      <group ref={group} position={[targetX, 0, 0]}>
        {shown.map((round, i) => (
          <group key={round}>
            <Block
              index={i}
              round={round}
              state={stateOf(round)}
              selected={selected === round}
              animate={animate}
              isNewest={i === visible - 1}
              onSelect={onSelect}
            />
            {i >= visible - LABELLED && (
              <Html position={[i * SPACING, -0.85, 0]} center distanceFactor={9} style={{ pointerEvents: "none" }}>
                <div className={`whitespace-nowrap font-mono text-[12px] ${stateOf(round) === "ok" ? "text-muted" : "text-failed"}`}>
                  R{round}
                  {stateOf(round) === "edited" && " - edited"}
                  {stateOf(round) === "broken" && " - link broken"}
                </div>
              </Html>
            )}
          </group>
        ))}
      </group>
    </group>
  );
}

export default function LedgerChain(props: {
  rounds: number[];
  visible: number;
  tamperedFrom: number | null;
  selected: number | null;
  onSelect: (round: number) => void;
  active: boolean;
  reduced: boolean;
}) {
  const animate = props.active && !props.reduced;
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [0, 1.8, 6.6], fov: 42 }}
      onCreated={({ camera }) => camera.lookAt(0, -0.2, 0)}
      frameloop={animate ? "always" : "demand"}
      gl={{ antialias: true, alpha: true }}
      fallback={
        <div className="flex h-full items-center justify-center p-6 text-center font-mono text-sm text-faint">
          3D view needs WebGL - the ledger is a chain of committed rounds.
        </div>
      }
      aria-label="3D view: a chain of ledger blocks, one per training round, each linked to the previous one by its hash."
    >
      <ambientLight intensity={1.0} />
      <directionalLight position={[4, 6, 5]} intensity={1.5} />
      <directionalLight position={[-5, 2, -3]} intensity={0.4} />
      <Chain {...props} animate={animate} />
    </Canvas>
  );
}
