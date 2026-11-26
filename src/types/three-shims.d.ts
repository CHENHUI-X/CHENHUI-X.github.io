declare module "three" {
	// biome-ignore lint/suspicious/noExplicitAny: three.js runtime module without @types/three
	const THREE: any;
	export = THREE;
}

declare module "three/examples/jsm/environments/RoomEnvironment.js" {
	export class RoomEnvironment {
		dispose?: () => void;
	}
}
