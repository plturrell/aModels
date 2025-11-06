declare module "*.module.css" {
  const classes: Record<string, string>;
  export default classes;
}

declare module "#repo/*?raw" {
  const contents: string;
  export default contents;
}
