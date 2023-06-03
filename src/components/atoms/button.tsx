// External Imports
import { motion } from "framer-motion";

type ButtonProps = {
  href?: string;
  children: React.ReactNode;
  target?: string;
  onClick?: () => void;
};

const Button = ({
  href,
  children,
  target,
  onClick,
}: ButtonProps): JSX.Element => {
  return (
    <a href={href} target={target} onClick={onClick}>
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        transition={{ duration: 0.1, ease: "easeOut" }}
        className="bg-gray-200 text-black hover:text-sky-600 rounded-xl p-4 text-lg font-bold cursor-pointer transition-colors duration-300 ease-in-out"
      >
        {children}
      </motion.div>
    </a>
  );
};

export default Button;
