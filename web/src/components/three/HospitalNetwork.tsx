"use client";

import { Html, Line, Sparkles } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export interface NetworkNode {
  id: number;
  images: number;
  highlight?: boolean;
}

const RING_RADIUS = 3.9;
const PARTICLES_PER_LINK = 7;
const HUB = new THREE.Vector3(0, 0, 0);

function layout(n: number): THREE.Vector3[] {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    return new THREE.Vector3(Math.cos(a) * RING_RADIUS, Math.sin(i * 1.9) * 0.7, Math.sin(a) * RING_RADIUS);
  });
}

/** The shared global model. */
function Hub({ animate }: { animate: boolean }) {
  const shell = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (animate && shell.current) {
      shell.current.rotation.y += delta * 0.18;
      shell.current.rotation.x += delta * 0.07;
    }
  });
  return (
    <group>
      <mesh ref={shell}>
        <icosahedronGeometry args={[0.95, 1]} />
        <meshBasicMaterial color="#2dd4bf" wireframe transparent opacity={0.45} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.46, 48, 48]} />
        <meshStandardMaterial color="#0b3b38" emissive="#2dd4bf" emissiveIntensity={0.9} roughness={0.3} />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.25, 32, 32]} />
        <meshBasicMaterial color="#2dd4bf" transparent opacity={0.04} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <pointLight color="#2dd4bf" intensity={25} distance={14} />
    </group>
  );
}

function HospitalNodeMesh({ node, position, radius, animate }: {
  node: NetworkNode;
  position: THREE.Vector3;
  radius: number;
  animate: boolean;
}) {
  const ring = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (animate && ring.current) {
      ring.current.rotation.z = state.clock.elapsedTime * 0.6;
      const s = 1 + Math.sin(state.clock.elapsedTime * 1.6) * 0.06;
      ring.current.scale.setScalar(s);
    }
  });
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial
          color="#e2e8f0"
          emissive={node.highlight ? "#99f6e4" : "#5eead4"}
          emissiveIntensity={node.highlight ? 0.9 : 0.35}
          roughness={0.35}
          metalness={0.1}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[radius * 1.9, 24, 24]} />
        <meshBasicMaterial color="#5eead4" transparent opacity={node.highlight ? 0.1 : 0.05} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      {node.highlight && (
        <mesh ref={ring} rotation={[Math.PI / 2.4, 0, 0]}>
          <torusGeometry args={[radius * 2.3, 0.018, 8, 96]} />
          <meshBasicMaterial color="#99f6e4" transparent opacity={0.8} />
        </mesh>
      )}
      <Html position={[0, radius + 0.38, 0]} center style={{ pointerEvents: "none" }}>
        <div className="whitespace-nowrap text-center font-mono">
          <div className={`text-[12px] font-semibold ${node.highlight ? "text-teal-200" : "text-slate-200"}`}>
            Centre {node.id}
          </div>
          <div className="text-[10px] text-slate-400">{node.images.toLocaleString("en-US")} images</div>
        </div>
      </Html>
    </group>
  );
}

/** Particles along each link: model updates in (bright), global model out (dim). */
function Flow({ positions, animate }: { positions: THREE.Vector3[]; animate: boolean }) {
  const n = positions.length * PARTICLES_PER_LINK;
  const upGeom = useRef<THREE.BufferGeometry>(null);
  const downGeom = useRef<THREE.BufferGeometry>(null);
  const up = useMemo(() => new Float32Array(n * 3), [n]);
  const down = useMemo(() => new Float32Array(n * 3), [n]);
  const tmp = useMemo(() => new THREE.Vector3(), []);

  useFrame((state) => {
    const t = animate ? state.clock.elapsedTime : 0;
    positions.forEach((p, i) => {
      for (let j = 0; j < PARTICLES_PER_LINK; j++) {
        const idx = (i * PARTICLES_PER_LINK + j) * 3;
        const u = (t * 0.16 + j / PARTICLES_PER_LINK + i * 0.11) % 1;
        tmp.lerpVectors(p, HUB, u);
        up[idx] = tmp.x;
        up[idx + 1] = tmp.y + Math.sin(u * Math.PI) * 0.35;
        up[idx + 2] = tmp.z;
        const v = (t * 0.11 + j / PARTICLES_PER_LINK + 0.5 + i * 0.07) % 1;
        tmp.lerpVectors(HUB, p, v);
        down[idx] = tmp.x;
        down[idx + 1] = tmp.y - Math.sin(v * Math.PI) * 0.25;
        down[idx + 2] = tmp.z;
      }
    });
    if (upGeom.current) upGeom.current.attributes.position.needsUpdate = true;
    if (downGeom.current) downGeom.current.attributes.position.needsUpdate = true;
  });

  return (
    <>
      <points>
        <bufferGeometry ref={upGeom}>
          <bufferAttribute attach="attributes-position" args={[up, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.09} color="#5eead4" transparent opacity={0.95} depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
      </points>
      <points>
        <bufferGeometry ref={downGeom}>
          <bufferAttribute attach="attributes-position" args={[down, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.06} color="#cbd5e1" transparent opacity={0.45} depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
      </points>
    </>
  );
}

/** Slight camera drift toward the pointer - window-level so text overlays don't block it. */
function CameraRig({ animate }: { animate: boolean }) {
  const pointer = useRef({ x: 0, y: 0 });
  useEffect(() => {
    if (!animate) return;
    const onMove = (e: PointerEvent) => {
      pointer.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      pointer.current.y = -((e.clientY / window.innerHeight) * 2 - 1);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, [animate]);

  useFrame(({ camera }) => {
    if (!animate) return;
    camera.position.x += (pointer.current.x * 1.5 - camera.position.x) * 0.03;
    camera.position.y += (3.8 + pointer.current.y * 0.8 - camera.position.y) * 0.03;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

function Scene({ nodes, animate }: { nodes: NetworkNode[]; animate: boolean }) {
  const group = useRef<THREE.Group>(null);
  const positions = useMemo(() => layout(nodes.length), [nodes.length]);
  const maxImages = Math.max(...nodes.map((n) => n.images));

  useFrame((_, delta) => {
    if (animate && group.current) group.current.rotation.y += delta * 0.045;
  });

  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[5, 8, 5]} intensity={0.8} />
      <CameraRig animate={animate} />
      <group ref={group} rotation={[0.08, 0.4, 0]}>
        <Hub animate={animate} />
        {nodes.map((node, i) => (
          <group key={node.id}>
            <Line
              points={[positions[i], HUB]}
              color={node.highlight ? "#99f6e4" : "#2dd4bf"}
              lineWidth={node.highlight ? 1.4 : 0.8}
              transparent
              opacity={node.highlight ? 0.55 : 0.25}
            />
            <HospitalNodeMesh
              node={node}
              position={positions[i]}
              radius={0.11 + 0.27 * Math.sqrt(node.images / maxImages)}
              animate={animate}
            />
          </group>
        ))}
        <Flow positions={positions} animate={animate} />
      </group>
      <Sparkles count={70} scale={[16, 8, 16]} size={1.6} speed={animate ? 0.25 : 0} opacity={0.35} color="#94a3b8" />
    </>
  );
}

export default function HospitalNetwork({ nodes, active, reduced }: {
  nodes: NetworkNode[];
  active: boolean;
  reduced: boolean;
}) {
  const animate = active && !reduced;
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [0, 3.8, 15], fov: 38 }}
      onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
      frameloop={animate ? "always" : "demand"}
      gl={{ antialias: true, alpha: true }}
      fallback={
        <div className="flex h-full items-center justify-center font-mono text-sm text-slate-500">
          3D view needs WebGL - six hospitals connected to one shared model.
        </div>
      }
      aria-label="3D view: six hospitals, sized by number of images, connected to a shared global model. Only model updates travel between them."
    >
      <Scene nodes={nodes} animate={animate} />
    </Canvas>
  );
}
